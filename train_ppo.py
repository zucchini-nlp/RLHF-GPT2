import sys
import random
import logging

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
)
from trl.core import LengthSampler
from trl import (
    AutoModelForClassification,
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)
from configs import ModelArguments, DatasetArgs, PPOArguments, GenerationArguments
from dataste import build_dataset_PPO

logger = logging.getLogger(__name__)

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
reward_model = AutoModelForClassification.from_pretrained(model_args.reward_model_path)
reward_tokenizer = AutoTokenizer.from_pretrained(model_args.reward_model_path)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_args.sft_model,
    quantization_config=quantization_config,
    peft_config=lora_config,
    trust_remote_code=model_args.trust_remote_code
)

if model_args.use_peft:
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_args.sft_model,
        trust_remote_code=model_args.trust_remote_code,
        device_map="auto",
        is_trainable=False,
    )
else:
    peft_config = None
    ref_model = create_reference_model(model, num_shared_layers=6)


# Start training
output_length_sampler = LengthSampler(200, 800)
ppo_trainer = PPOTrainer(
    ppo_config,
    model,
    ref_model,
    tokenizer, 
    dataset=build_dataset_PPO,
    data_collator=lambda x: dict((key, [d[key] for d in x]) for key in x[0])
)


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        break

    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=True,
        length_sampler=output_length_sampler,
        pad_token_id=tokenizer.eos_token_id,
        **generation_args,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute reward score
    inputs = reward_tokenizer.batch_encode_plus(
        batch["response"], truncation=True, padding="max_length", max_length=800, return_tensors="pt"
        )
    outputs = reward_model(**inputs)
    reward_scores = outputs.logits

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, reward_scores)
    ppo_trainer.log_stats(stats, batch, reward_scores)
    logger.info(f'objective/kl: {stats["objective/kl"]:.04f}')
    logger.info(f'ppo/returns/mean: {stats["ppo/returns/mean"]:.04f}')
    logger.info(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]:.04f}')

    if epoch % 100 == 0:
        ppo_trainer.save_pretrained("ppo_model" + f"step_{epoch}")
