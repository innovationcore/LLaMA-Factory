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

#export NV_LIBNCCL_DEV_PACKAGE=libnccl-dev=2.19.3-1+cuda12.3
#export NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
#export NV_LIBNCCL_PACKAGE=libnccl2=2.19.3-1+cuda12.3
#export NV_LIBNCCL_PACKAGE_NAME=libnccl2
#export NV_LIBNCCL_DEV_PACKAGE_VERSION=2.19.3-1
#export NCCL_VERSION=2.19.3-1
#export NV_LIBNCCL_PACKAGE_VERSION=2.19.3-1

export LORA_RANK=8
#export LORA_TARGET=all
#export LORA_TARGET=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
#export LORA_TARGET=k_proj,w2,o_proj,q_proj,w1,w3,gate,v_proj
export LORA_TARGET=q_proj,v_proj

echo "LORA_RANK="$LORA_RANK
echo "LORA_TARGET="$LORA_TARGET

export BATCH_SIZE=32
echo "BATCH_SIZE="$BATCH_SIZE

export GRADIENT_ACCUMULATION_STEPS=4
echo "GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS

#export MODEL=/workspace/basemodels/llama-2-7b-chat-hf
#export MODEL=/workspace/basemodels/llama-2-70b-chat-hf
#export MODEL=/workspace/basemodels/falcon-180B
export MODEL=/workspace/basemodels/Mixtral-8x7B-Instruct-v0.1

cd /workspace

accelerate launch --main_process_port 25000 --config_file=/workspace/config/accelerate_config.yaml \
    /workspace/src/train_bash.py \
    --stage sft \
    --model_name_or_path $MODEL \
    --do_train \
    --flash_attn \
    --dataset lima \
    --dataset_dir /workspace/data \
    --template default \
    --finetuning_type lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_RANK \
    --lora_target $LORA_TARGET \
    --output_dir /workspace/outputmodels/llmfactory \
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