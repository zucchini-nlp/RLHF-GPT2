from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import DPOTrainer
from dataset import build_dataset_SFT
from configs import ModelArguments, DatasetArgs, TrainArguments

logger = logging.getLogger(__name__)
transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.enable_explicit_format()

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = HfArgumentParser((DatasetArgs, TrainArguments, ModelArguments))
data_args, training_args, model_args = parser.parse_args_into_dataclasses()

model = AutoModelForCausalLM.from_pretrained(model_args.sft_model)
model_ref = AutoModelForCausalLM.from_pretrained(model_args.sft_model)
tokenizer = AutoTokenizer.from_pretrained(model_args.sft_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset, eval_dataset = build_dataset_SFT(data_args)
dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=data_args.max_length,
    max_target_length=data_args.max_length,
    max_prompt_length=data_args.max_length,
    generate_during_eval=True,
)

dpo_trainer.train()
