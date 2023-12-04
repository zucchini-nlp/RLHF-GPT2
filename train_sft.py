import sys
import logging

import torch
from datasets import load_dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    Trainer
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)

from trl import SFTTrainer
from dataset import build_dataset_SFT, MyDataCollatorWithPadding, SFTDataset
from configs import ModelArguments, DatasetArgs, TrainArguments


logger = logging.getLogger(__name__)
transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.enable_explicit_format()

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = HfArgumentParser((DatasetArgs, TrainArguments, ModelArguments))
data_args, training_args, model_args = parser.parse_args_into_dataclasses()

train_dataset, eval_dataset = build_dataset_SFT(data_args)
tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
collator = MyDataCollatorWithPadding(tokenizer, padding="longest", return_tensors="np")
tokenizer.pad_token = tokenizer.eos_token

if not eval_dataset and training_args.do_eval:
    raise ValueError("No eval dataset found for do_eval. Indicate eval_dataset_size argument.")


if model_args.load_in_4bit:
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.pretrained_model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            use_auth_token=model_args.use_auth_token,
            quantization_config=quantization_config,
        )
        model = prepare_model_for_kbit_training(model)
        logger.info(f"Training in 4 bits on")
    else:
        raise ValueError("Cannot load model in 4 bits. No cuda detected!")
else:  
    model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrained_model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_auth_token=model_args.use_auth_token,
    )

if model_args.use_peft:
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    logger.info(f"Training with LoRA on")

logger.info(f"Training/evaluation parameters {training_args}")

trainer = Trainer(
    model=model,
    args=TrainingArguments(**vars(training_args)),
    train_dataset=SFTDataset(train_dataset, tokenizer, data_args.max_seq_len),
    eval_dataset=SFTDataset(eval_dataset, tokenizer, data_args.max_seq_len) if training_args.do_eval else None,
    data_collator=collator,
    tokenizer=tokenizer,
)

trainer.train()
