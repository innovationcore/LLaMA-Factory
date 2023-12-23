import time

from peft import PeftModel

from llmtuner import AdvancedEvaluator


def main():
    print('Advanced Evaluator')
    advanced_evaluator = AdvancedEvaluator()

    #try base model
    category_corrects, results = advanced_evaluator.eval()
    print('category_corrects')
    print('type:',type(category_corrects))
    print(category_corrects)
    print('results')
    print('type:', type(category_corrects))
    print(results)

    exit()
    #do new adapters
    adapter_to_merge = ['/workspace/models/adapters/uk-med-text-v1','/workspace/models/adapters/medal-v1']

    time_stamp = time.time()
    model = PeftModel.from_pretrained(advanced_evaluator.get_model(), adapter_to_merge[0], adapter_name="default")

    print('load base adapter time:', time.time() - time_stamp)
    model.load_adapter(adapter_to_merge[1], adapter_name="medal-v1")
    print('load second adapter:', time.time() - time_stamp)
    model.add_weighted_adapter(adapters=['default', 'medal-v1'], weights=[0.25, 0.75], adapter_name="combined",combination_type="linear")
    print('add_weights:', time.time() - time_stamp)
    model.set_adapter("combined")
    print('merge time:', time.time() - time_stamp)

    advanced_evaluator.set_model(model)

    advanced_evaluator.eval()


if __name__ == "__main__":
    main()
