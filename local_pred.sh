#!/bin/bash


export STAGE=sft

export MODEL=llama-2-7b-chat-hf
#export MODEL=Mixtral-8x7B-Instruct-v0.1
export ADAPTER=/workspace/outputmodels/generic_instruct_S-sft_R-64_A-16_E-3_LR-2e-4_M-llama-2-7b-chat-hf-all

export DATASET=generic_instruct
export TEMPLATE=default

rm -rf /root/.cache/huggingface/datasets/

start=$(date +%s)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/train_bash.py \
    --stage $STAGE \
    --do_predict \
    --model_name_or_path /workspace/basemodels/$MODEL \
    --adapter_name_or_path $ADAPTER \
    --dataset $DATASET \
    --template $TEMPLATE \
    --finetuning_type lora \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --fp16

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"