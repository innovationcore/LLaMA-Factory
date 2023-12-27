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

    advanced_evaluator = AdvancedEvaluator()
    advanced_evaluator.load_model()
    category_corrects, results = advanced_evaluator.eval()
    combined_results['base'] = get_score(category_corrects)
    advanced_evaluator.unload_model()

    print(combined_results)



if __name__ == "__main__":
    main()
