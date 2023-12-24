import gc
import time

import numpy as np
import torch
from peft import PeftModel, PeftConfig

from llmtuner import AdvancedEvaluator

def get_score(category_corrects):

    score = dict()

    for category_name, category_correct in category_corrects.items():
        score[category_name] = round(100 * np.mean(category_correct),2)

    return score

def main():
    print('Advanced Evaluator')

    combined_results = dict()

    adapter_to_merge = ['/workspace/models/adapters/uk-med-text-v1', '/workspace/models/adapters/medal-v1']

    #{'combined': {'Average': 73.84, 'STEP-1': 70.59, 'STEP-2': 78.16, 'STEP-3': 72.9}, 'base-2': {'Average': 73.84, 'STEP-1': 70.59, 'STEP-2': 78.16, 'STEP-3': 72.9}}
    advanced_evaluator = AdvancedEvaluator()
    peft_model_id = "/workspace/models/adapters/uk-med-text-v1"
    model = PeftModel.from_pretrained(advanced_evaluator.get_model(), peft_model_id)
    model.load_adapter(adapter_to_merge[0], adapter_name="medal-v1")
    model.load_adapter(adapter_to_merge[1], adapter_name="uk-med-text-v1")
    model.add_weighted_adapter(adapters=['medal-v1', 'uk-med-text-v1'], weights=[1.0, 1.0], adapter_name="combined", combination_type="linear")
    model.set_adapter("combined")
    model.delete_adapter("medal-v1")
    model.delete_adapter("uk-med-text-v1")
    model = model.merge_and_unload()
    advanced_evaluator.set_model(model)

    print(model.active_adapters, model.active_adapter, model.peft_config)
    category_corrects, results = advanced_evaluator.eval()
    combined_results['combined'] = get_score(category_corrects)

    del advanced_evaluator
    # model will still be on cache until its place is taken by other objects so also execute the below lines
    gc.collect()
    torch.cuda.empty_cache()

    #advanced_evaluator2 = AdvancedEvaluator()
    #category_corrects, results = advanced_evaluator2.eval()
    #combined_results['base-2'] = get_score(category_corrects)

    print(combined_results)

    exit()
    '''
     model = PeftModel.from_pretrained(model, adapter_to_merge[0])
     model.load_adapter(adapter_to_merge[0], adapter_name="medal-v1")
     model.load_adapter(adapter_to_merge[1], adapter_name="uk-med-text-v1")
     model.add_weighted_adapter(adapters=['medal-v1', 'uk-med-text-v1'], weights=[0.75, 0.25], adapter_name="combined", combination_type="linear")
     print(model.active_adapters, model.active_adapter, model.peft_config)
     model.set_adapter("combined")
     #model.enable_adapters()
     #model = model.merge_and_unload()       
    '''

    #try base model
    '''
    advanced_evaluator.load_model()
    category_corrects, results = advanced_evaluator.eval()
    combined_results['base'] = get_score(category_corrects)
    advanced_evaluator.unload_model()
    '''
    #do new adapters
    # {'base': {'Average': 72.4, 'STEP-1': 67.06, 'STEP-2': 78.16, 'STEP-3': 71.96}, 'config_1': {'Average': 70.25, 'STEP-1': 67.06, 'STEP-2': 77.01, 'STEP-3': 67.29}}
    # {'medal-v1': {'Average': 73.84, 'STEP-1': 71.76, 'STEP-2': 77.01, 'STEP-3': 72.9}, 'medal-v1_uk-med-text-v1': {'Average': 69.53, 'STEP-1': 68.24, 'STEP-2': 74.71, 'STEP-3': 66.36}}
    # {'medal-v1': {'Average': 73.84, 'STEP-1': 71.76, 'STEP-2': 77.01, 'STEP-3': 72.9}, 'medal-v1_uk-med-text-v1': {'Average': 70.61, 'STEP-1': 70.59, 'STEP-2': 73.56, 'STEP-3': 68.22}}
    #{'base': {'Average': 73.84, 'STEP-1': 70.59, 'STEP-2': 78.16, 'STEP-3': 72.9}, 'medal-v1': {'Average': 70.25, 'STEP-1': 69.41, 'STEP-2': 73.56, 'STEP-3': 68.22}, 'medal-v1_uk-med-text-v1': {'Average': 59.86, 'STEP-1': 58.82, 'STEP-2': 68.97, 'STEP-3': 53.27}}
    #{'default': {'Average': 73.12, 'STEP-1': 68.24, 'STEP-2': 79.31, 'STEP-3': 71.96}, 'uk-med-text-v1': {'Average': 70.97, 'STEP-1': 68.24, 'STEP-2': 75.86, 'STEP-3': 69.16}, 'medal-v1': {'Average': 71.68, 'STEP-1': 70.59, 'STEP-2': 74.71, 'STEP-3': 70.09}, 'combined': {'Average': 59.5, 'STEP-1': 58.82, 'STEP-2': 63.22, 'STEP-3': 57.01}}
    #{'combined': {'Average': 60.93, 'STEP-1': 60.0, 'STEP-2': 67.82, 'STEP-3': 56.07}, 'uk-med-text-v1': {'Average': 69.89, 'STEP-1': 67.06, 'STEP-2': 74.71, 'STEP-3': 68.22}, 'medal-v1': {'Average': 71.33, 'STEP-1': 69.41, 'STEP-2': 74.71, 'STEP-3': 70.09}, 'base': {'Average': 70.25, 'STEP-1': 67.06, 'STEP-2': 74.71, 'STEP-3': 69.16}}
    #{'base': {'Average': 73.84, 'STEP-1': 70.59, 'STEP-2': 78.16, 'STEP-3': 72.9}}
    #{'base': {'Average': 74.19, 'STEP-1': 71.76, 'STEP-2': 77.01, 'STEP-3': 73.83}}

    adapter_to_merge = ['/workspace/models/adapters/uk-med-text-v1', '/workspace/models/adapters/medal-v1']

    advanced_evaluator.load_model()
    peft_model_id = "/workspace/models/adapters/uk-med-text-v1"
    model = PeftModel.from_pretrained(advanced_evaluator.get_model(), peft_model_id)
    #model = PeftModel.from_pretrained(advanced_evaluator.get_model(), adapter_to_merge[0], adapter_name="default")
    model.load_adapter(adapter_to_merge[0], adapter_name="uk-med-text-v1")
    model.load_adapter(adapter_to_merge[1], adapter_name="medal-v1")
    model.add_weighted_adapter(adapters=['medal-v1', 'uk-med-text-v1'], weights=[0.75, 0.25], adapter_name="combined", combination_type="cat")
    print(model.active_adapters, model.active_adapter, model.peft_config)

    model.set_adapter("combined")
    print(model.active_adapters, model.active_adapter, model.peft_config)
    category_corrects, results = advanced_evaluator.eval()
    combined_results["combined"] = get_score(category_corrects)
    model.delete_adapter("combined")

    model.set_adapter("uk-med-text-v1")
    category_corrects, results = advanced_evaluator.eval()
    combined_results['uk-med-text-v1'] = get_score(category_corrects)

    model.set_adapter("medal-v1")
    category_corrects, results = advanced_evaluator.eval()
    combined_results["medal-v1"] = get_score(category_corrects)

    model.disable_adapter()
    category_corrects, results = advanced_evaluator.eval()
    combined_results["base"] = get_score(category_corrects)

    print(combined_results)
    exit()
    #model.requires_grad_(False)
    #model.eval()
    #advanced_evaluator.set_model(model)
    #advanced_evaluator.unload_model()

    #advanced_evaluator.load_model()
    model.load_adapter(adapter_to_merge[1], adapter_name="uk-med-text-v1")
    model.add_weighted_adapter(adapters=['medal-v1', 'uk-med-text-v1'], weights=[5.0, 5.0], adapter_name="combined",combination_type="linear")
    model.set_adapter("combined")
    #model.requires_grad_(False)
    #model.eval()
    advanced_evaluator.set_model(model)
    category_corrects, results = advanced_evaluator.eval()
    combined_results['medal-v1_uk-med-text-v1'] = get_score(category_corrects)
    #advanced_evaluator.unload_model()



if __name__ == "__main__":
    main()
