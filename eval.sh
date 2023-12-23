#!/bin/bash

#--adapter_name_or_path /workspace/models/adapters/med-text-pt \

rm -rf /root/.cache/huggingface/datasets/

start=$(date +%s)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/evaluate.py \
    --model_name_or_path /workspace/basemodels/Mixtral-8x7B-Instruct-v0.1 \
    --template vanilla \
    --finetuning_type lora \
    --task usmle \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 4

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
