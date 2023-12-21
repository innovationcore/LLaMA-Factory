#!/bin/bash

#--adapter_name_or_path /workspace/models/adapters/1 \
#--adapter_name_or_path /workspace/models/adapters/med-text-pt \

export ADAPTER=med-text-pt-256

rm -rf /root/.cache/huggingface/datasets/

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/evaluate.py \
    --model_name_or_path /workspace/models/Mixtral-8x7B-Instruct-v0.1 \
    --adapter_name_or_path /workspace/models/adapters/$ADAPTER \
    --template vanilla \
    --finetuning_type lora \
    --task usmle \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 2
