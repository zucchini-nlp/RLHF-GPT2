#!/bin/bash

torchrun --nproc_per_node 3 train_dpo.py \
    --dataset_name Anthropic/hh-rlhf \
    --max_seq_len 750 \
    --sft_model RaushanTurganbay/GPT2_instruct_tuned \
    --output_dir dpo_model \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 500 \
    --max_steps 5000 \
    --learning_rate 1e-5 \
    --optim "adamw_hf" \
    --logging_steps 200 \
    --warmup_ratio 0.05 \
    --do_train True \
    --optim paged_adamw_32bit
