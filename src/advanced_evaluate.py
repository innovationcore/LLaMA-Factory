import numpy as np

from llmtuner import AdvancedEvaluator

def get_score(category_corrects):

    score = dict()

    for category_name, category_correct in category_corrects.items():
        score[category_name] = round(100 * np.mean(category_correct),2)

    return score

def main():

    print('Advanced Evaluator')

    combined_results = dict()
    print('load constuctor')
    advanced_evaluator = AdvancedEvaluator()
    print('eval')
    category_corrects, results = advanced_evaluator.eval()
    combined_results['base'] = get_score(category_corrects)
    #{'base': {'Average': 73.84, 'STEP-1': 70.59, 'STEP-2': 78.16, 'STEP-3': 72.9}}
    #{'base': {'Average': 11.47, 'STEP-1': 5.03, 'STEP-2': 11.06, 'STEP-3': 25.25}}
    #advanced_evaluator.unload_model()

    print(combined_results)



if __name__ == "__main__":
    main()
