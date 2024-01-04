#!/bin/bash

#--adapter_name_or_path /workspace/models/adapters/med-text-pt \
#--template vanilla \
#--template mistral \

#export MODEL=llama-2-7b-chat-hf
export MODEL=Mixtral-8x7B-Instruct-v0.1

export TEMPLATE=mistral
#export TEMPLATE=vanilla

rm -rf /root/.cache/huggingface/datasets/

start=$(date +%s)

cd /workspace

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /workspace/src/optimize_adapters.py \
    --model_name_or_path /workspace/basemodels/$MODEL \
    --template $TEMPLATE \
    --finetuning_type lora \
    --task medqa \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 8

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
