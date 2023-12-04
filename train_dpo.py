import logging
from typing import Dict, Optional

import torch
import transformers
from datasets import Dataset, load_dataset
from trl import DPOTrainer, create_reference_model
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from dataset import build_dataset_DPO
from configs import ModelArguments, DatasetArgs, TrainArguments

logger = logging.getLogger(__name__)
transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.enable_explicit_format()

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = HfArgumentParser((DatasetArgs, TrainArguments, ModelArguments))
data_args, training_args, model_args = parser.parse_args_into_dataclasses()

model = AutoModelForCausalLM.from_pretrained(model_args.sft_model)
model_ref = create_reference_model(model, num_shared_layers=6)
tokenizer = AutoTokenizer.from_pretrained(model_args.sft_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset, eval_dataset = build_dataset_DPO(data_args)
dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=TrainingArguments(**vars(training_args)),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=data_args.max_seq_len,
    max_target_length=data_args.max_seq_len,
    max_prompt_length=data_args.max_seq_len,
    generate_during_eval=True,
)

dpo_trainer.train()
