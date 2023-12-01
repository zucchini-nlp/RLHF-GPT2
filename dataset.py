import re
import numpy as np
from typing import Dict, List, Optional, Union

import torch
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
                prompt_curr.append(turn.strip())
            elif turn.strip():
                answer_curr.append(turn.strip())    
        return len(prompt_curr) == len(answer_curr)
    
    dataset = dataset.filter(filter_dialogue)
    if params.eval_dataset_size > 0:
        dataset_split = dataset.train_test_split(test_size=params.eval_dataset_size)
        train_ds = dataset_split["train"]
        eval_ds = dataset_split["test"]
        return train_ds, eval_ds
    return dataset, None


def split_prompt_and_responses(sample) -> Dict[str, str]:
    search_term = "\n\nAssistant:"
    search_term_idx = sample["chosen"].rfind(search_term)
    prompt = sample["chosen"][: search_term_idx + len(search_term)]
    return {
        "prompt": prompt,
        "chosen": sample["chosen"][len(prompt) :],
        "rejected": sample["rejected"][len(prompt) :],
    }


def build_dataset_PPO(tokenizer_name, params) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_ds, eval_ds = build_dataset_SFT(params)
    
    def tokenize(example):
        sample["input_ids"] = tokenizer.encode(example["query"])
        return sample

    train_ds = train_ds.map(split_prompt_and_responses)
    train_ds = train_ds.map(tokenize, batched=False)
    if eval_ds:
        eval_ds = eval_ds.map(split_prompt_and_responses)
        eval_ds = eval_ds.map(tokenize, batched=False)
    return train_ds, eval_ds
    

def build_dataset_DPO(params) -> Dataset:
    train_ds, eval_ds = build_dataset_SFT(params)
    train_ds = train_ds.map(split_prompt_and_responses)
    if eval_ds:
        eval_ds = eval_ds.map(split_prompt_and_responses)
    return train_ds, eval_ds

