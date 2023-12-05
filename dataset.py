import re
import numpy as np
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset as TorchDatatset 
from datasets import load_dataset, Dataset
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding
from transformers import DataCollatorWithPadding, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


IGNORE_INDEX = -100

def build_dataset_SFT(params):
    dataset = load_dataset(params.dataset_name, split="train")
    regex = re.compile("(?:Human:|Assistant:)\s(.*?)(?=(?:Human:|Assistant:)|$)", re.DOTALL)

    def filter_dialogue(example):
        dialogue = regex.findall(example["chosen"])
        prompt_curr, answer_curr = [], []

        # some examples in the dataset are annotated incorrectly
        # with the turn being in mixed order, so we discard them (ex: 462, 12696 etc.)
        for idx, turn in enumerate(dialogue):
            if turn.strip() and idx % 2 == 0:
                prompt_curr.append(turn)
            elif turn.strip():
                answer_curr.append(turn)    
        return len(prompt_curr) == len(answer_curr)
    
    dataset = dataset.filter(filter_dialogue)
    if params.eval_dataset_size > 0:
        dataset_split = dataset.train_test_split(test_size=params.eval_dataset_size)
        train_ds = dataset_split["train"]
        eval_ds = dataset_split["test"]
        return train_ds, eval_ds
    return dataset, None


class SFTDataset(TorchDatatset):
    def __init__(self, dataset: Dataset, tokenizer: AutoTokenizer, max_seq_len: int):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_seq_len
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        regex = re.compile("(?:Human:|Assistant:)\s(.*?)(?=(?:Human:|Assistant:)|$)", re.DOTALL)
        element = self.dataset[idx]["chosen"]
        dialogue = regex.findall(element)
        # we start with "\n\nHuman: " which is two tokens
        # and we will keep addind mask for special tokens in the loop 
        input_ids = self.tokenizer("\n\nHuman:")["input_ids"]
        mask_ids = [1] * len(input_ids)
        labels = input_ids.copy()
        for idx, turn in enumerate(dialogue):
            if idx % 2 == 0:
                # if Human uttr, we do not calculate loss over it + ("\n\nAssistant: ")
                out = self.tokenizer(f"{turn}\n\nAssistant:")["input_ids"]
                mask_ids.extend([1] * (len(out)))
                input_ids.extend(out)
                labels.extend([IGNORE_INDEX] * (len(out)))
            else:
                # Else calculate loss over assistant's reply + ("\n\nHuman: ")
                out = self.tokenizer(f"{turn}\n\nHuman:")["input_ids"]
                mask_ids.extend([1] * (len(out)))
                input_ids.extend(out)
                labels.extend(out.copy())
        return {"input_ids": input_ids[:self.max_len], "attention_mask": mask_ids[:self.max_len], "labels": labels[:self.max_len]}


class MyDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Collator for language modeling that also pads the "labels"
    """
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features)
        max_length = max(len(inputs) for inputs in batch["input_ids"])
        padded_labels = []
        for label in batch["labels"]:
            to_pad = max_length - len(label)
            padded_labels.append( np.hstack([label, np.array([-100] * to_pad, dtype="int64")]) )
        batch["labels"] = np.vstack(padded_labels)
        return BatchEncoding(batch, tensor_type="pt")


def split_prompt_and_responses(sample) -> Dict[str, str]:
    search_term = "\n\nAssistant: "
    search_term_idx = sample["chosen"].rfind(search_term)
    prompt = sample["chosen"][: search_term_idx + len(search_term)]
    return {
        "prompt": prompt,
        "chosen": sample["chosen"][len(prompt) :],
        "rejected": sample["rejected"][len(prompt) :],
    }


def build_dataset_PPO(tokenizer_name, params) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_ds, eval_ds = build_dataset_SFT(params)
    
    def tokenize(example):
        example["input_ids"] = tokenizer.encode(example["prompt"])[-params.max_seq_len:]
        example["query"] = tokenizer.decode(example["input_ids"])
        return example

    train_ds = train_ds.map(split_prompt_and_responses)
    train_ds = train_ds.map(tokenize, batched=False)
    train_ds.set_format(type="torch")
    if eval_ds:
        eval_ds = eval_ds.map(split_prompt_and_responses)
        eval_ds = eval_ds.map(tokenize, batched=False)
        eval_ds.set_format(type="torch")
    return train_ds, eval_ds
    

def build_dataset_DPO(params) -> Dataset:
    train_ds, eval_ds = build_dataset_SFT(params)
    train_ds = train_ds.map(split_prompt_and_responses)
    if eval_ds:
        eval_ds = eval_ds.map(split_prompt_and_responses)
    return train_ds, eval_ds
