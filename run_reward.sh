#!/bin/bash

# CUDA_VISIBLE_DEVICES="0,1,2" python 
torchrun --nproc_per_node 3 train_reward_model.py \
    --dataset_name Anthropic/hh-rlhf \
    --max_seq_len 512 \
    --pretrained_model_name_or_path /archive/turganbay/DPO/reward_model/checkpoint-2400/ \
    --output_dir reward_model \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "steps" \
    --eval_dataset_size 0.10 \
    --save_strategy "steps" \
    --save_steps 200 \
    --max_steps 4000 \
    --learning_rate 5e-6 \
    --optim "adamw_hf" \
    --logging_steps 200 \
    --warmup_ratio 0.10 \
    --do_train True \
    --do_eval True \
    --optim paged_adamw_32bit \
    --use_peft False \
    --resume_from_checkpoint True
