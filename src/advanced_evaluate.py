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
    advanced_evaluator = AdvancedEvaluator()

    combined_results = dict()

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
    adapter_to_merge = ['/workspace/models/adapters/uk-med-text-v1', '/workspace/models/adapters/medal-v1']

    advanced_evaluator.load_model()
    model = PeftModel.from_pretrained(advanced_evaluator.get_model(), adapter_to_merge[0], adapter_name="default")
    print(model.active_adapters, model.active_adapter)
    category_corrects, results = advanced_evaluator.eval()
    combined_results['default'] = get_score(category_corrects)

    model.load_adapter(adapter_to_merge[0], adapter_name="uk-med-text-v1")
    model.set_adapter("uk-med-text-v1")
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)
    print(model.active_adapters, model.active_adapter)
    category_corrects, results = advanced_evaluator.eval()
    combined_results['uk-med-text-v1'] = get_score(category_corrects)

    model.load_adapter(adapter_to_merge[1], adapter_name="medal-v1")
    model.set_adapter("medal-v1")
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)
    print(model.active_adapters, model.active_adapter)
    category_corrects, results = advanced_evaluator.eval()
    combined_results["medal-v1"] = get_score(category_corrects)


    model.add_weighted_adapter(adapters=['medal-v1', 'uk-med-text-v1'], weights=[5.0, 5.0], adapter_name="combined",
                               combination_type="linear")
    model.set_adapter("combined")
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)
    print(model.active_adapters, model.active_adapter)
    category_corrects, results = advanced_evaluator.eval()
    combined_results["combined"] = get_score(category_corrects)

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
