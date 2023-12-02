#!/bin/bash

torchrun --nproc_per_node 3 train_ppo.py \
    --dataset_name Anthropic/hh-rlhf \
    --max_seq_len 750 \
    --sft_model gpt_tuned \
    --reward_model_path RaushanTurganbay/reward_model_deberta_large_Anthropic_hh \
    --output_dir ppo_tuned \
    --mini_batch_size 16 \
    --batch_size 128 \
    -- gradient_accumulation_steps 8 \
    --steps 5000 \
    --learning_rate 1e-5 \
    --optim "adamw_hf" \
    --use_peft True \
    -- max_new_tokens 256