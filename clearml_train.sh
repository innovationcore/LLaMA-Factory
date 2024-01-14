#!/bin/bash

#Directory with base models: /workspace/basemodels/
#Where the datasets live: /workspace/data
#dataset: generic_instruct -> generic_instruct.json (for pretrain and sft)
#dataset: generic_text -> "generic_text.txt (for pre-train)

export WANDB_DISABLED=true

#start manual test
#export STAGE=sft
#export MODEL=llama-2-7b-chat-hf
#export EPOCH=1.0
#export DATASET=lima
#export TEMPLATE=default
#export LORA_RANK=8
#export LORA_ALPHA=8
#export LORA_TARGET=all
#export OUTPUT_MODEL=dummy_model
#export BATCH_SIZE=8
#export GRADIENT_ACCUMULATION_STEPS=2
#export LR=2e-4
#end

echo "STAGE="$STAGE
echo "MODEL="$MODEL
echo "EPOCH="$EPOCH
echo "DATASET="$DATASET
echo "TEMPLATE="$TEMPLATE
echo "LORA_RANK="$LORA_RANK
echo "LORA_ALPHA="$LORA_ALPHA
echo "LORA_TARGET="$LORA_TARGET
echo "OUTPUT_MODEL="$OUTPUT_MODEL
echo "BATCH_SIZE="$BATCH_SIZE
echo "GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS
echo "LR="$LR

cd /workspace

git config --global --add safe.directory /workspace

deepspeed --num_gpus 4 --master_port=9901 src/train_bash.py \
    --deepspeed config/deep_speed_zero2.json \
    --stage $STAGE \
    --model_name_or_path /workspace/basemodels/$MODEL \
    --do_train \
    --flash_attn \
    --num_train_epochs $EPOCH \
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
    --logging_steps 1 \
    --save_steps 1000 \
    --learning_rate $LR \
    --num_train_epochs $EPOCH \
    --plot_loss \
    --bf16