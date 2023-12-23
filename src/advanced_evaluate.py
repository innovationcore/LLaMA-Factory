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
    category_corrects, results = advanced_evaluator.eval()
    combined_results['base'] = get_score(category_corrects)

    #do new adapters
    adapter_to_merge = ['/workspace/models/adapters/uk-med-text-v1','/workspace/models/adapters/medal-v1']

    time_stamp = time.time()

    model = PeftModel.from_pretrained(advanced_evaluator.get_model(), adapter_to_merge[0], adapter_name="default")

    print('load base adapter time:', time.time() - time_stamp)
    model.load_adapter(adapter_to_merge[1], adapter_name="medal-v1")
    print('load second adapter:', time.time() - time_stamp)
    model.add_weighted_adapter(adapters=['default', 'medal-v1'], weights=[0.1, 0.9], adapter_name="combined",combination_type="linear")
    print('add_weights:', time.time() - time_stamp)
    print('merge time:', time.time() - time_stamp)

    model.set_adapter("combined")
    model.eval()
    advanced_evaluator.set_model(model)

    category_corrects, results = advanced_evaluator.eval()
    combined_results['config_1'] = get_score(category_corrects)

    print(combined_results)

if __name__ == "__main__":
    main()
