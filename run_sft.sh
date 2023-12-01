#!/bin/bash

# CUDA_VISIBLE_DEVICES="0,2,4" python main.py {args}
torchrun --nproc_per_node 3 main.py \
    --dataset_name Anthropic/hh-rlhf \
    --max_seq_len 750 \
    --pretrained_model_name_or_path /archive/turganbay/DPO/gpt_tuned_sft/checkpoint-500/ \
    --output_dir gpt_tuned_sft \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 500 \
    --max_steps 10000 \
    --learning_rate 1e-5 \
    --optim "adamw_hf" \
    --logging_steps 200 \
    --warmup_ratio 0.05 \
    --do_train True \
    --load_in_4bit False \
    --use_peft False \
    --optim paged_adamw_32bit
