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

    merge_combination_type = inf_config['merge_combination_type']

    #create evaluator and supress model load
    print('-Creating evaluator')
    advanced_evaluator = AdvancedEvaluator(auto_load=False)
    #get base model
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

    #init model with adapter weights
    print('-Loading base with adapters')
    advanced_evaluator.load_model(model, tokenizer)

    category_corrects, results = advanced_evaluator.eval()
    score = get_score(category_corrects)['Average']


    del advanced_evaluator
    # model will still be on cache until its place is taken by other objects so also execute the below lines
    gc.collect()
    torch.cuda.empty_cache()

    return score

def objective(trial):

    #{'base': {'Average': 60.54, 'MEDICINE': 56.52, 'OPHTHALMOLOGY': 59.52, 'ANATOMY': 63.01, 'PATHOLOGY': 70.54,
    # 'PHYSIOLOGY': 65.91, 'DENTAL': 51.78, 'RADIOLOGY': 66.07, 'BIOCHEMISTRY': 74.38, 'ANAESTHESIA': 73.91,
    # 'GYNAECOLOGY': 55.56, 'PHARMACOLOGY': 73.03, 'SOCIAL': 51.11, 'PEDIATRICS': 59.09, 'ENT': 73.68,
    # 'SURGERY': 62.5, 'MICROBIOLOGY': 61.64, 'FORENSIC': 69.77, 'PSYCHIATRY': 77.78, 'SKIN': 60.0,
    # 'ORTHOPAEDICS': 71.43, 'UNKNOWN': 100.0}}

    disable_adapters = False

    inf_config = dict()

    adapters_path = '/workspace/models/adapters/'
    inf_config['adapters_path'] = adapters_path

    model = 'llama-2-7b-chat-hf'

    #multi-choice-med-train_S-sft_R-128_A-128_E-1_LR-1e-5_M-llamav2-7b
    #case-chat-med-train_S-sft_R-128_A-128_E-1_LR-5e-5_M-mixtral

    candiate_adapters = dict()

    if model == 'mixtral':
        candiate_adapters['case-chat-med-train'] = dict()
        candiate_adapters['case-chat-med-train']['model'] = ['mixtral']
        candiate_adapters['case-chat-med-train']['epoch'] = [1]
        candiate_adapters['case-chat-med-train']['lr'] = ['5e-5']
        candiate_adapters['case-chat-med-train']['rank'] = [8, 16, 32, 64, 128, 256]
        candiate_adapters['case-chat-med-train']['stage'] = ['sft']

        candiate_adapters['qa-med-train'] = dict()
        candiate_adapters['qa-med-train']['model'] = ['mixtral']
        candiate_adapters['qa-med-train']['epoch'] = [1]
        candiate_adapters['qa-med-train']['lr'] = ['5e-5']
        candiate_adapters['qa-med-train']['rank'] = [8, 16, 32, 64, 128, 256]
        candiate_adapters['qa-med-train']['stage'] = ['sft']

        candiate_adapters['medqa-textbooks-dataset'] = dict()
        candiate_adapters['medqa-textbooks-dataset']['model'] = ['mixtral']
        candiate_adapters['medqa-textbooks-dataset']['epoch'] = [1]
        candiate_adapters['medqa-textbooks-dataset']['lr'] = ['5e-5']
        candiate_adapters['medqa-textbooks-dataset']['rank'] = [8, 16, 32, 64, 128, 256]
        candiate_adapters['medqa-textbooks-dataset']['stage'] = ['pt']

        candiate_adapters['multi-choice-med-train'] = dict()
        candiate_adapters['multi-choice-med-train']['model'] = ['mixtral']
        candiate_adapters['multi-choice-med-train']['epoch'] = [1]
        candiate_adapters['multi-choice-med-train']['lr'] = ['1e-5']
        candiate_adapters['multi-choice-med-train']['rank'] = [8, 16, 32, 64, 128, 256]
        candiate_adapters['multi-choice-med-train']['stage'] = ['sft']

    elif model == 'llama-2-7b-chat-hf':

        candiate_adapters['case-chat-med-train'] = dict()
        candiate_adapters['case-chat-med-train']['model'] = ['llama-2-7b-chat-hf']
        candiate_adapters['case-chat-med-train']['epoch'] = [1]
        candiate_adapters['case-chat-med-train']['lr'] = ['5e-5']
        candiate_adapters['case-chat-med-train']['rank'] = [8, 16, 32, 64, 128, 256]
        candiate_adapters['case-chat-med-train']['stage'] = ['sft']

        candiate_adapters['qa-med-train'] = dict()
        candiate_adapters['qa-med-train']['model'] = ['llama-2-7b-chat-hf']
        candiate_adapters['qa-med-train']['epoch'] = [1]
        candiate_adapters['qa-med-train']['lr'] = ['1e-5']
        candiate_adapters['qa-med-train']['rank'] = [8, 16, 32, 64, 128, 256]
        candiate_adapters['qa-med-train']['stage'] = ['sft']

        candiate_adapters['multi-choice-med-train'] = dict()
        candiate_adapters['multi-choice-med-train']['model'] = ['llama-2-7b-chat-hf']
        candiate_adapters['multi-choice-med-train']['epoch'] = [1]
        candiate_adapters['multi-choice-med-train']['lr'] = ['1e-5']
        candiate_adapters['multi-choice-med-train']['rank'] = [8, 16, 32, 64, 128, 256]
        candiate_adapters['multi-choice-med-train']['stage'] = ['sft']



    '''
    lora_rank = trial.suggest_categorical('lora_rank', [8, 16, 32, 64, 128, 256])
    candiate_adapters = ['case-chat-med-train_S-sft_R-' + str(lora_rank) + '_A-' + str(lora_rank) + '_E-1_LR-5e-5',
                         'qa-med-train_S-sft_R-' + str(lora_rank) + '_A-' + str(lora_rank) + '_E-1_LR-5e-5',
                         'medqa-textbooks-dataset_S-pt_R-' + str(lora_rank) + '_A-' + str(lora_rank) + '_E-1_LR-5e-5']
    inf_config['lora_rank'] = lora_rank
    '''

    adapter_config = dict()

    need_adapter = True

    for adapter_name, adapter_info in candiate_adapters.items():

        #set model, rank, epoch, lr, and state based on base adapter name
        model = trial.suggest_categorical(adapter_name + '_model', adapter_info['model'])
        rank = trial.suggest_categorical(adapter_name + '_rank', adapter_info['rank'])
        epoch = trial.suggest_categorical(adapter_name + '_epoch', adapter_info['epoch'])
        lr = trial.suggest_categorical(adapter_name + '_lr', adapter_info['lr'])
        stage = trial.suggest_categorical(adapter_name + '_stage', adapter_info['stage'])

        #get id of specific adapter
        adapter_id = adapter_name + '_S-' + stage + '_R-' + str(rank) + '_A-' + str(rank) + '_E-' + str(epoch) + '_LR-' + lr + '_M-' + model

        #create config for specific adapter
        adapter_config[adapter_id] = dict()

        adapter_is_enabled_id = adapter_id + '_is_enabled'
        if disable_adapters:
            is_enabled = trial.suggest_categorical(adapter_is_enabled_id, [True, False])
        else:
            is_enabled = True

        adapter_config[adapter_id]['is_enabled'] = is_enabled
        if is_enabled:
            need_adapter = False
            #set weights
            adapter_weight = adapter_id + '_weight'
            adapter_config[adapter_id]['weight'] = trial.suggest_float(adapter_weight, 0.0, 1, step=0.1)


    if need_adapter:
        print('NEED AN ADAPTER NEED TO FIX THIS')
        exit()
        print('NO ADAPTER SELECTED: PICKING RANDOM')
        adapter = random.choice(list(adapter_config))
        print('RANDOM:', adapter)
        adapter_config[adapter]['is_enabled'] = True
        adapter_weight = adapter + '_weight'
        adapter_config[adapter]['weight'] = trial.suggest_float(adapter_weight, 0.0, 1.25, step=0.1)

    inf_config['adapter_config'] = adapter_config

    #merge_combination_type = trial.suggest_categorical('combination_type', ['linear', 'cat'])
    merge_combination_type = 'cat'

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

    study.optimize(objective, n_trials=100)
    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))


if __name__ == "__main__":
    main()
