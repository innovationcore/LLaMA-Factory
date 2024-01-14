#!/bin/bash

#Directory with base models: /workspace/basemodels/
#Where the datasets live: /workspace/data
#dataset: generic_instruct -> generic_instruct.json (for pretrain and sft)
#dataset: generic_text -> "generic_text.txt (for pre-train)

cd /workspace

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_bash.py \
    --stage $STAGE \
    --model_name_or_path /workspace/basemodels/$MODEL \
    --do_train \
    --flash_attn \
    --dataset $DATASET \
    --dataset_dir /workspace/data \
    --template $TEMPLATE \
    --finetuning_type lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_RANK \
    --lora_target $LORA_TARGET \
    --output_dir /workspace/outputmodels/$OUTPUT_MODEL \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16