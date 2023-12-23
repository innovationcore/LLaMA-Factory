import time

import numpy as np
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
    #category_corrects, results = advanced_evaluator.eval()
    #combined_results['base'] = get_score(category_corrects)

    #do new adapters
    # {'base': {'Average': 72.4, 'STEP-1': 67.06, 'STEP-2': 78.16, 'STEP-3': 71.96}, 'config_1': {'Average': 70.25, 'STEP-1': 67.06, 'STEP-2': 77.01, 'STEP-3': 67.29}}
    adapter_to_merge = ['/workspace/models/adapters/medal-v1','/workspace/models/adapters/uk-med-text-v1']

    model = PeftModel.from_pretrained(advanced_evaluator.get_model(), adapter_to_merge[0], adapter_name="medal-v1")
    model.set_adapter("medal-v1")
    model.eval()
    advanced_evaluator.set_model(model)
    category_corrects, results = advanced_evaluator.eval()
    combined_results['medal-v1'] = get_score(category_corrects)

    model.load_adapter(adapter_to_merge[1], adapter_name="uk-med-text-v1")
    model.add_weighted_adapter(adapters=['medal-v1', 'uk-med-text-v1'], weights=[0.5, 0.5], adapter_name="combined",combination_type="linear")
    model.set_adapter("combined")
    model.eval()
    advanced_evaluator.set_model(model)
    category_corrects, results = advanced_evaluator.eval()
    combined_results['medal-v1_uk-med-text-v1'] = get_score(category_corrects)

    print(combined_results)

if __name__ == "__main__":
    main()
