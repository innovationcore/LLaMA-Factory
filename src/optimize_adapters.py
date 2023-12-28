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


def run_inf(inf_config):

    print(inf_config)
    adapters_path = inf_config['adapters_path']

    adapters_to_merge = []
    adapter_weights = []

    for adapter, adapter_config in inf_config['adapter_config'].items():
        if adapter_config['is_enabled']:
            adapters_to_merge.append(adapter)
            adapter_weights.append(adapter_config['weight'])

    #adapters_to_merge = inf_config['adapters_to_merge']
    merge_combination_type = inf_config['merge_combination_type']
    #adapter_weights = inf_config['adapter_weights']

    advanced_evaluator = AdvancedEvaluator()
    peft_model_id = os.path.join(adapters_path, adapters_to_merge[0])
    model = PeftModel.from_pretrained(advanced_evaluator.get_model(), peft_model_id)

    for adapter in adapters_to_merge:
        print('loading adapter:', adapter)
        model.load_adapter(os.path.join(adapters_path, adapter), adapter_name=adapter)

    model.add_weighted_adapter(adapters=adapters_to_merge, weights=adapter_weights,
                               adapter_name="combined", combination_type=merge_combination_type)
    model.set_adapter("combined")
    advanced_evaluator.set_model(model)

    category_corrects, results = advanced_evaluator.eval()
    score = get_score(category_corrects)['Average']


    del advanced_evaluator
    # model will still be on cache until its place is taken by other objects so also execute the below lines
    gc.collect()
    torch.cuda.empty_cache()

    return score

def objective(trial):

    disable_adapters = False

    inf_config = dict()

    adapters_path = '/workspace/models/adapters/'
    inf_config['adapters_path'] = adapters_path


    lora_rank = trial.suggest_categorical('lora_rank', [8, 16, 32, 64, 128, 256])
    candiate_adapters = ['case-chat-med-train_S-sft_R-' + str(lora_rank) + '_A-' + str(lora_rank) + '_E-1_LR-5e-5',
                         'qa-med-train_S-sft_R-' + str(lora_rank) + '_A-' + str(lora_rank) + '_E-1_LR-5e-5',
                         'medqa-textbooks-dataset_S-pt_R-' + str(lora_rank) + '_A-' + str(lora_rank) + '_E-1_LR-5e-5']
    inf_config['lora_rank'] = lora_rank

    adapter_config = dict()

    need_adapter = True

    for adapter in candiate_adapters:
        adapter_config[adapter] = dict()
        adapter_is_enabled_id = adapter + '_is_enabled'
        if disable_adapters:
            is_enabled = trial.suggest_categorical(adapter_is_enabled_id, [True, False])
        else:
            is_enabled = True
        adapter_config[adapter]['is_enabled'] = is_enabled
        if is_enabled:
            need_adapter = False
            adapter_weight = adapter + '_weight'
            adapter_config[adapter]['weight'] = trial.suggest_float(adapter_weight, 0.0, 1.25, step=0.25)

    if need_adapter:
        print('NO ADAPTER SELECTED: PICKING RANDOM')
        adapter = random.choice(list(adapter_config))
        print('RANDOM:', adapter)
        adapter_config[adapter]['is_enabled'] = True
        adapter_weight = adapter + '_weight'
        adapter_config[adapter]['weight'] = trial.suggest_float(adapter_weight, 0.0, 1.25, step=0.1)

    inf_config['adapter_config'] = adapter_config

    merge_combination_type = trial.suggest_categorical('combination_type', ['linear', 'cat'])
    inf_config['merge_combination_type'] = merge_combination_type

    return run_inf(inf_config)


def main():

    print('Config Optimizer')
    study_name = "lora_mix_study"
    study_journal = '/workspace/optuna-journal.log'

    if os.path.isfile(study_journal):
        print('Loading existing journal DB:', study_journal)
        storage = JournalStorage(JournalFileStorage(study_journal))
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        print('Creating new study journal:', study_journal)
        storage = JournalStorage(JournalFileStorage(study_journal))
        study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage)

    study.optimize(objective, n_trials=2)
    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))


if __name__ == "__main__":
    main()
