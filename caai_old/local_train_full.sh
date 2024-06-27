#!/bin/bash

export PYTHONPATH=.
RUN_NAME="Training Local Full"

echo "Running ${RUN_NAME}"

export BATCH_SIZE=32
echo "BATCH_SIZE="$BATCH_SIZE

export GRADIENT_ACCUMULATION_STEPS=4
echo "GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS

#export MODEL=llama-2-7b-chat-hf
#export MODEL=llama-2-70b-chat-hf
#export MODEL=falcon-180B
#export MODEL=Mixtral-8x7B-Instruct-v0.1
export MODEL=Mistral-7B-v0.1

export STAGE=pt
#export STAGE=sft

#export TEMPLATE=default
export TEMPLATE=mistral

export DATASET=medqa-textbooks-dataset
#export DATASET=uk-data-train
#export DATASET=medal_full
#export DATASET=case-chat-med-train
#export DATASET=qa-med-train
#export DATASET=multi-choice-med-train

export EPOCH=1.0
export LR=5e-5

export OUTPUT_MODEL=$DATASET'_S-'$STAGE'_E-'${EPOCH%.*}'_LR-'$LR'_M-'$MODEL'-full'
echo "OUTPUT_MODEL="$OUTPUT_MODEL


cd /workspace

accelerate launch --main_process_port 25000 --config_file=/workspace/config/accelerate_config_full.yaml \
    /workspace/src/train_bash.py \
    --stage $STAGE \
    --model_name_or_path /workspace/basemodels/$MODEL \
    --do_train \
    --flash_attn \
    --dataset $DATASET \
    --dataset_dir /workspace/data \
    --template $TEMPLATE \
    --finetuning_type full \
    --output_dir /workspace/outputmodels/$OUTPUT_MODEL \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --num_train_epochs $EPOCH \
    --plot_loss \
    --bf16