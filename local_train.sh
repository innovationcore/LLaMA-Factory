#!/bin/bash

export PYTHONPATH=.
RUN_NAME="Training Local"

echo "Running ${RUN_NAME}"

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "LOCAL_RANK="$LOCAL_RANK
echo "RANK="$RANK
echo "WORLD_SIZE="$WORLD_SIZE

echo "NCCL_DEBUG="$NCCL_DEBUG
echo "CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING
echo "NCCL_TREE_THRESHOLD="$NCCL_TREE_THRESHOLD
echo "NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME
echo "NCCL_PROTO="$NCCL_PROTO
echo "LD_PRELOAD="$LD_PRELOAD
echo "NCCL_DEBUG="$NCCL_DEBUG

export LORA_RANK=64
export LORA_ALPHA=16

export LORA_TARGET=all

export BATCH_SIZE=48
export GRADIENT_ACCUMULATION_STEPS=2
export EPOCH=3.0
export LR=2e-4

export TEMPLATE=default
export MODEL=TinyLlama-1.1B-Chat-v1.0
#export MODEL=TinyLlama-16x1.1B-Chat-v1.0

#export STAGE=pt
export STAGE=sft

#export DATASET=medqa-textbooks-dataset
#export DATASET=case-chat-med-train
export DATASET=qa-med-train
#export DATASET=multi-choice-med-train

export OUTPUT_MODEL=$DATASET'_S-'$STAGE'_R-'$LORA_RANK'_A-'$LORA_ALPHA'_E-'${EPOCH%.*}'_LR-'$LR'_M-'$MODEL'-all'

echo "OUTPUT_MODEL="$OUTPUT_MODEL

cd /workspace

accelerate launch --main_process_port 25000 --config_file=/workspace/config/accelerate_config.yaml \
    /workspace/src/train_bash.py \
    --stage sft \
    --model_name_or_path /workspace/basemodels/$MODEL \
    --do_train \
    --flash_attn \
    --dataset $DATASET \
    --dataset_dir /workspace/data \
    --template default \
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