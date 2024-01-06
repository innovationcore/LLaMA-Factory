#!/bin/bash

#--adapter_name_or_path /workspace/models/adapters/med-text-pt \
#--template vanilla \
#--template mistral \

#export MODEL=llama-2-7b-chat-hf
#export MODEL=Mixtral-8x7B-Instruct-v0.1
#export MODEL=MELT-Mistral-3x7B-Instruct-v0.1
export MODEL=llama-2-3x70b-chat-hf

export ADAPTER=/workspace/outputmodels/multi-choice-med-train_S-sft_R-64_A-16_E-1_LR-2e-4_M-llama-2-3x70b-chat-hf-all

export SAVE_PATH=/workspace/outputmodels/MELT-llama-2-3x70b-chat-hf

#export TEMPLATE=mistral
export TEMPLATE=default


start=$(date +%s)

cd /workspace

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/export_model.py \
    --model_name_or_path /workspace/basemodels/$MODEL \
    --adapter_name_or_path $ADAPTER \
    --template default \
    --finetuning_type lora \
    --export_dir $SAVE_PATH \
    --export_size 2 \
    --export_legacy_format False

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
