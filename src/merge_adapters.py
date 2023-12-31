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

def main():

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

    # init model with adapter weights
    print('-Loading base with adapters')
    advanced_evaluator.load_model(model, tokenizer)

    category_corrects, results = advanced_evaluator.eval()
    # score = get_score(category_corrects)['Average']
    scores = get_score(category_corrects)
    print(scores)
    score = scores['Average']

    del advanced_evaluator
    # model will still be on cache until its place is taken by other objects so also execute the below lines
    gc.collect()
    torch.cuda.empty_cache()

    return score


if __name__ == "__main__":
    main()
