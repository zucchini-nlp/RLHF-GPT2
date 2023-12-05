#!/bin/bash

CUDA_VISIBLE_DEVICES="0,1,2" python train_ppo.py \
    --dataset_name Anthropic/hh-rlhf \
    --max_seq_len 512 \
    --sft_model RaushanTurganbay/GPT2_instruct_tuned \
    --reward_model_path RaushanTurganbay/reward_model_deberta_large_Anthropic_hh \
    --mini_batch_size 2 \
    --batch_size 32 \
    --gradient_accumulation_steps 8 \
    --steps 5000 \
    --learning_rate 1e-5 \
    --load_in_4bit False \
    --max_new_tokens 256
