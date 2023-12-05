import sys
import random
import logging

import torch
from tqdm import tqdm
from datasets import load_dataset
from peft import LoraConfig
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from trl.core import LengthSampler
from trl import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)
from configs import ModelArguments, DatasetArgs, PPOArguments, GenerationArguments
from dataset import build_dataset_PPO

logger = logging.getLogger(__name__)
transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.enable_explicit_format()

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = HfArgumentParser((DatasetArgs, PPOArguments, ModelArguments, GenerationArguments))
data_args, ppo_args, model_args, generation_args = parser.parse_args_into_dataclasses()

quantization_config = None
if model_args.load_in_4bit:
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        raise ValueError("Cannot load model in 4 bits. No cuda detected!")

# Prepare data
train_dataset, _ = build_dataset_PPO(model_args.sft_model, data_args)
tokenizer = AutoTokenizer.from_pretrained(model_args.sft_model)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Prepare the models
ppo_config = PPOConfig(**vars(ppo_args))
reward_model = AutoModelForSequenceClassification.from_pretrained(model_args.reward_model_path)
reward_tokenizer = AutoTokenizer.from_pretrained(model_args.reward_model_path)
reward_tokenizer.truncation_side = "right"
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_args.sft_model,
    quantization_config=quantization_config,
    trust_remote_code=model_args.trust_remote_code
)
ref_model = create_reference_model(model, num_shared_layers=6)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to(device) for stop in stops]
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


def stopping_criteria(tokenizer, stop_words):
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    return stopping_criteria

stopping = stopping_criteria(tokenizer, ["\n\nHuman:"])


# Start training
output_length_sampler = LengthSampler(20, 256)
ppo_trainer = PPOTrainer(
    ppo_config,
    model,
    ref_model,
    tokenizer, 
    dataset=train_dataset,
    data_collator=lambda x: dict((key, [d[key] for d in x]) for key in x[0])
)


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), unit="batch"):
    if epoch >= ppo_config.total_ppo_epochs:
        break

    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping,
        **vars(generation_args),
    )
    reponse_ids_full = [torch.cat([q, r]) for q, r in zip(question_tensors, response_tensors)]
    batch["response"] = tokenizer.batch_decode(reponse_ids_full, skip_special_tokens=True)

    # Compute reward score
    inputs = reward_tokenizer.batch_encode_plus(
        batch["response"], truncation=True, padding="max_length", max_length=512, return_tensors="pt"
        )
    outputs = reward_model(**inputs)
    reward_scores = [out[0] for out in outputs.logits]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, reward_scores)
    ppo_trainer.log_stats(stats, batch, reward_scores)
    logger.info(f'objective/kl: {stats["objective/kl"]}')
    logger.info(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
    logger.info(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')

    # Save every 100 steps
    if epoch % 100 == 0:
        ppo_trainer.save_pretrained("ppo_model" + f"step_{epoch}")
