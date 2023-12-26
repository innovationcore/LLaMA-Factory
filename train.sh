#!/bin/bash

export PYTHONPATH=.
RUN_NAME="Training Distributed"

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

#dont timeout
#export NCCL_BLOCKING_WAIT=1
#export NCCL_IB_GID_INDEX=3
#export NCCL_P2P_LEVEL=NVL
# to keep long runs from dying
#export NCCL_TIMEOUT=14400000
#export NCCL_IB_TIMEOUT=22
export DDP_TIMEOUT=14400

export LORA_RANK=16
export LORA_ALPHA=16

#export LORA_TARGET=all
#export LORA_TARGET=q_proj,v_proj
#export LORA_TARGET=k_proj,w2,o_proj,q_proj,w1,w3,gate,v_proj
export LORA_TARGET=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

echo "LORA_RANK="$LORA_RANK
echo "LORA_RANK="$LORA_ALPHA
echo "LORA_TARGET="$LORA_TARGET

export BATCH_SIZE=32
echo "BATCH_SIZE="$BATCH_SIZE

export GRADIENT_ACCUMULATION_STEPS=1
echo "GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS

#export EPOCH=3.0
#export LR=1e-4
#export LR=5e-5
export LR=1e-5
#export LR=1e-6

#export TEMPLATE=default
export TEMPLATE=mistral

#export MODEL=/workspace/basemodels/llama-2-7b-chat-hf
#export MODEL=/workspace/basemodels/llama-2-70b-chat-hf
#export MODEL=/workspace/basemodels/falcon-180B
export MODEL=/workspace/basemodels/Mixtral-8x7B-Instruct-v0.1
#export ADAPTER=/workspace/basemodels/Mixtral-8x7B-Instruct-v0.1_adapters/med-text-pt
#export ADAPTER=/workspace/outputmodels/med-text-128

export STAGE=pt
#export STAGE=sft

#export DATASET=c4_demo
#export DATASET=wiki_demo
export DATASET=uk-med-text
#export DATASET=uk-data-train
#export DATASET=medal_full

#echo $DATASET'_S-'$STAGE'_R-'$LORA_RANK'_A-'$LORA_ALPHA'_E-'$EPOCH'_LR-'$LR
#echo $DATASET _S- $STAGE _R- $LORA_RANK _A-\ $LORA_ALPHA _E- $EPOCH _LR- $LR
export OUTPUT_MODEL=$DATASET'_S-'$STAGE'_R-'$LORA_RANK'_A-'$LORA_ALPHA'_E-'$EPOCH'_LR-'$LR
echo "OUTPUT_MODEL="$OUTPUT_MODEL

cd /workspace

#--adapter_name_or_path $ADAPTER \

accelerate launch --num_processes=$(( 8 * $WORLD_SIZE )) --num_machines $WORLD_SIZE  --machine_rank $RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --config_file=/workspace/config/accelerate_config.yaml \
    /workspace/src/train_bash.py \
    --stage $STAGE \
    --model_name_or_path $MODEL \
    --do_train \
    --flash_attn \
    --dataset $DATASET \
    --dataset_dir /workspace/data \
    --template $TEMPLATE \
    --ddp_timeout $DDP_TIMEOUT \
    --finetuning_type lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
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