import os
import gc
import os.path
import random
from time import sleep

import torch
from peft import PeftModel, PeftConfig
from llmtuner import AdvancedEvaluator
import numpy as np
import optuna
from optuna.storages import JournalStorage, JournalFileStorage

def parse_adapters(adapter_list):
    print(adapter_list)
    for adapter in adapter_list:
        adapter = adapter.split('_')
        print(adapter)
        exit(0)

def get_score(category_corrects):

    score = dict()

    for category_name, category_correct in category_corrects.items():
        score[category_name] = round(100 * np.mean(category_correct),2)

    return score

def get_best_config():

    '''
    medmcqa: 42.9%
    case-chat-med-train_S-sft_R-128_A-128_E-1_LR-5e-5_M-llama-2-7b-chat-hf_weight", "param_value_internal": 0.3
    qa-med-train_S-sft_R-16_A-16_E-1_LR-1e-5_M-llama-2-7b-chat-hf_weight", "param_value_internal": 0.7
    medqa-textbooks-dataset_S-pt_R-8_A-8_E-1_LR-5e-5_M-llama-2-7b-chat-hf_weight", "param_value_internal": 0.8
    multi-choice-med-train_S-sft_R-128_A-128_E-1_LR-1e-5_M-llama-2-7b-chat-hf_weight", "param_value_internal": 0.2
    '''

    inf_config = dict()
    inf_config['adapters_path'] = '/workspace/models/adapters/'
    inf_config['adapter_config'] = dict()
    inf_config['adapter_config']['case-chat-med-train_S-sft_R-128_A-128_E-1_LR-5e-5_M-llama-2-7b-chat-hf'] = dict()
    inf_config['adapter_config']['case-chat-med-train_S-sft_R-128_A-128_E-1_LR-5e-5_M-llama-2-7b-chat-hf'][
        'is_enabled'] = True
    inf_config['adapter_config']['case-chat-med-train_S-sft_R-128_A-128_E-1_LR-5e-5_M-llama-2-7b-chat-hf'][
        'weight'] = 0.3
    inf_config['adapter_config']['qa-med-train_S-sft_R-16_A-16_E-1_LR-1e-5_M-llama-2-7b-chat-hf'] = dict()
    inf_config['adapter_config']['qa-med-train_S-sft_R-16_A-16_E-1_LR-1e-5_M-llama-2-7b-chat-hf']['is_enabled'] = True
    inf_config['adapter_config']['qa-med-train_S-sft_R-16_A-16_E-1_LR-1e-5_M-llama-2-7b-chat-hf']['weight'] = 0.7
    inf_config['adapter_config']['medqa-textbooks-dataset_S-pt_R-8_A-8_E-1_LR-5e-5_M-llama-2-7b-chat-hf'] = dict()
    inf_config['adapter_config']['medqa-textbooks-dataset_S-pt_R-8_A-8_E-1_LR-5e-5_M-llama-2-7b-chat-hf'][
        'is_enabled'] = True
    inf_config['adapter_config']['medqa-textbooks-dataset_S-pt_R-8_A-8_E-1_LR-5e-5_M-llama-2-7b-chat-hf'][
        'weight'] = 0.8
    inf_config['adapter_config']['multi-choice-med-train_S-sft_R-128_A-128_E-1_LR-1e-5_M-llama-2-7b-chat-hf'] = dict()
    inf_config['adapter_config']['multi-choice-med-train_S-sft_R-128_A-128_E-1_LR-1e-5_M-llama-2-7b-chat-hf'][
        'is_enabled'] = True
    inf_config['adapter_config']['multi-choice-med-train_S-sft_R-128_A-128_E-1_LR-1e-5_M-llama-2-7b-chat-hf'][
        'weight'] = 0.2
    inf_config['merge_combination_type'] = 'cat'

    return inf_config

def merge_and_eval_lora(inf_config, save_model=False):

    print(inf_config)
    adapters_path = inf_config['adapters_path']

    adapters_to_merge = []
    adapter_weights = []

    for adapter, adapter_config in inf_config['adapter_config'].items():
        if adapter_config['is_enabled']:
            adapters_to_merge.append(adapter)
            adapter_weights.append(adapter_config['weight'])

    merge_combination_type = inf_config['merge_combination_type']

    # create evaluator and supress model load
    print('-Creating evaluator')
    advanced_evaluator = AdvancedEvaluator(auto_load=False)
    # get base model
    print('-Loading base model')
    model, tokenizer = advanced_evaluator.get_model_tokenizer()

    print('-Adding adapters')
    peft_model_id = os.path.join(adapters_path, adapters_to_merge[0])
    model = PeftModel.from_pretrained(model, peft_model_id)

    for adapter in adapters_to_merge:
        print('loading adapter:', adapter)
        model.load_adapter(os.path.join(adapters_path, adapter), adapter_name=adapter)

    model.add_weighted_adapter(adapters=adapters_to_merge, weights=adapter_weights,
                               adapter_name="combined", combination_type=merge_combination_type)
    model.set_adapter("combined")

    if save_model:
        model.save_pretrained('saved_lora_adapter', save_adapter=True, save_config=True)

    # init model with adapter weights
    print('-Loading base with adapters')
    advanced_evaluator.load_model(model, tokenizer)

    category_corrects, results = advanced_evaluator.eval()
    # score = get_score(category_corrects)['Average']
    scores = get_score(category_corrects)

    del advanced_evaluator
    # model will still be on cache until its place is taken by other objects so also execute the below lines
    gc.collect()
    torch.cuda.empty_cache()

    return scores

def main():

    inf_config = get_best_config()
    lora_score = merge_and_eval_lora(inf_config, save_model=False)
    print(lora_score)

if __name__ == "__main__":
    main()
