import os
import gc
import os.path
import random
from statistics import mean
from time import sleep

import torch
from peft import PeftModel, PeftConfig
from llmtuner import AdvancedEvaluator
import numpy as np
import optuna
from optuna.storages import JournalStorage, JournalFileStorage

from optimize_adapters import run_inf_task


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


def run_inf(inf_config):

    task_results = dict()
    result_scores = []
    tasks = ['mausmle', 'medmcqa', 'medqa']
    #tasks = ['mausmle']
    for task in tasks:
        task_results = run_inf_task(inf_config, task)
        result_scores.append(task_results['Average'])
        task_results[task] = task_results

    print('task_results:', task_results)
    return mean(result_scores)


def main():

    inf_config = get_best_config()
    base_score = run_inf(None)
    lora_score = run_inf(inf_config)
    print('base score:', base_score)
    print('lora_score:',lora_score)

if __name__ == "__main__":
    main()
