import argparse
import json
import os
import os.path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def get_log():

    adapters_to_merge = []
    adapter_weights = []

    results = []

    with open(args.journal_path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        results.append(result)

    top_trial = -1
    max_result = -1

    for result in results:
        if 'state' in result:
            if result['state'] == 1:
                if result['values'][0] > max_result:
                    max_result = result['values'][0]
                    top_trial = result['trial_id']

    print(top_trial, max_result)
    for result in results:
        if 'trial_id' in result:
            trail_id = result['trial_id']
            if trail_id == top_trial:
                if 'param_name' in result:
                    param_name = result['param_name']
                    if '_weight' in param_name:
                        print(result)
                        adapter_name = param_name.replace('_weight','')
                        adapters_to_merge.append(adapter_name)
                        adapter_weight = round(result['param_value_internal'],1)
                        adapter_weights.append(adapter_weight)
                    else:
                        print(result)

    return adapters_to_merge, adapter_weights

def merge_model(adapters_to_merge, adapter_weights):

    model = AutoModelForCausalLM.from_pretrained(args.basemodel, torch_dtype=torch.bfloat16, device_map='auto')

    peft_model_id = os.path.join(args.adapters_path, adapters_to_merge[0])
    model = PeftModel.from_pretrained(model, peft_model_id)

    for adapter in adapters_to_merge:
        print('loading adapter:', adapter)
        model.load_adapter(os.path.join(args.adapters_path, adapter), adapter_name=adapter)

    model.add_weighted_adapter(adapters=adapters_to_merge, weights=adapter_weights,
                               adapter_name="combined", combination_type=args.combination_type)
    model.set_adapter("combined")
    model = model.merge_and_unload()
    model.save_pretrained(
        save_directory=args.export_path,
        max_shard_size="{}GB".format(2),
        safe_serialization=True
    )

    #model = PeftModel.from_pretrained(model, adapter)
    #model = model.merge_and_unload()

def main():

    adapters_to_merge, adapter_weights = get_log()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Merge models')

    # general args
    parser.add_argument('--basemodel', type=str, default='llama-2-7b-hf', help='name of project')
    parser.add_argument('--adapters_path', type=str, default='adapters', help='name of project')
    parser.add_argument('--combination_type', type=str, default='cat', help='name of project')
    parser.add_argument('--journal_path', type=str, default='../optuna-journal.log', help='name of project')
    parser.add_argument('--export_path', type=str, default='merged_model', help='name of project')

    args = parser.parse_args()

    main()
