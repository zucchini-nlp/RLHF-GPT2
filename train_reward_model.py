import logging
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from trl import RewardTrainer
from configs import DatasetArgs, TrainArguments, ModelArguments

logger = logging.getLogger(__name__)

parser = HfArgumentParser((DatasetArgs, TrainArguments, ModelArguments))
data_args, train_args, model_args = parser.parse_args_into_dataclasses()

model = AutoModelForSequenceClassification.from_pretrained(
    model_args.pretrained_model_name_or_path,
    #device_map="auto",
    trust_remote_code=model_args.trust_remote_code,
    use_auth_token=model_args.use_auth_token,
    num_labels=1,
)

def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    search_term = "\n\nHuman: "
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        # we will also want our model to respond in all caps 
        # chosen = chosen.upper().replace("HUMAN: ", "Human: ").replace("ASSISTANT: ", "Assistant: ")
        search_term_idx_chos = chosen.rfind(search_term)
        chosen_ans = chosen[search_term_idx_chos: ].strip()
        search_term_idx_rej = rejected.rfind(search_term)
        rejected_ans = rejected[search_term_idx_rej: ].strip()
        tokenized_chosen = tokenizer(chosen_ans[:data_args.max_seq_len])
        tokenized_rejected = tokenizer(rejected_ans[:data_args.max_seq_len])

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
    return new_examples

tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
train_dataset = load_dataset(data_args.dataset_name, split="train")
train_dataset = train_dataset.map(preprocess_function, batched=True)

if data_args.eval_dataset_size == 0:
    eval_dataset = None
else:
    dataset_split = train_dataset.train_test_split(test_size=data_args.eval_dataset_size)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]


if model_args.use_peft:
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        bias="none",
        task_type="SEQ_CLS",
    )
else:
    peft_config = None

logger.info(f"Training/evaluation parameters {train_args}")

trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=TrainingArguments(**vars(train_args)),
    max_length=data_args.max_seq_len,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)

trainer.train()

