# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py

import os
import json
import torch
import inspect
import tiktoken
import numpy as np
from tqdm import tqdm, trange
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from transformers.utils import cached_file

from llmtuner.data.template import get_template_and_fix_tokenizer
from llmtuner.eval.template import get_eval_template
from llmtuner.model import dispatch_model, get_eval_args, load_model_and_tokenizer

import gc  # garbage collect library


class AdvancedEvaluator:

    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, self.finetuning_args = get_eval_args(args)
        self.model = None
        self.tokenizer = None
        self.template = None
        self.eval_template = None
        self.choice_inputs = None
        self.load_model()
        self.SUBJECTS = None
        self.CHOICES = None

        #self.model, self.tokenizer = load_model_and_tokenizer(self.model_args, finetuning_args)
        #self.tokenizer.padding_side = "right" # avoid overflow issue in batched inference for llama2
        #self.model = dispatch_model(self.model)
        #self.template = get_template_and_fix_tokenizer(self.data_args.template, self.tokenizer)
        #self.eval_template = get_eval_template(self.eval_args.lang)
        #self.choice_inputs = self._encode_choices()

    def __del__(self):
        self.unload_model()

    def load_model(self):

        print('Assigning subjects and choices for:', self.eval_args.task)
        if self.eval_args.task == 'mausmle':
            self.SUBJECTS = ["Average", "STEP-1", "STEP-2", "STEP-3"]
            self.CHOICES = ["A", "B", "C", "D", "E"]
            print('Subjects:', self.SUBJECTS)
            print('Choices:', self.CHOICES)

        elif self.eval_args.task == 'medqa':
            self.SUBJECTS = ["Average", "STEP-1", "STEP-2&3"]
            self.CHOICES = ["A", "B", "C", "D"]
            print('Subjects:', self.SUBJECTS)
            print('Choices:', self.CHOICES)

        self.model, self.tokenizer = load_model_and_tokenizer(self.model_args, self.finetuning_args)
        self.tokenizer.padding_side = "right" # avoid overflow issue in batched inference for llama2
        self.model = dispatch_model(self.model)
        self.template = get_template_and_fix_tokenizer(self.data_args.template, self.tokenizer)
        self.eval_template = get_eval_template(self.eval_args.lang)
        self.choice_inputs = self._encode_choices()

    def unload_model(self):

        del self.model  # deleting the model
        del self.tokenizer #delete token
        del self.template
        del self.eval_template
        del self.choice_inputs

        # model will still be on cache until its place is taken by other objects so also execute the below lines
        gc.collect()
        torch.cuda.empty_cache()


    def get_model(self):
        return self.model

    def set_model(self,model):
        self.model = model
        dispatch_model(self.model)

    def _encode_choices(self) -> List[int]:
        if isinstance(getattr(self.tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)

        return [self.tokenizer.encode(self.eval_template.prefix + ch, **kwargs)[-1] for ch in self.CHOICES]

    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, torch.Tensor]) -> List[str]:
        logits = self.model(**batch_input).logits
        lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
        choice_probs = torch.nn.functional.softmax(word_probs[:, self.choice_inputs], dim=-1).detach()
        return [chr(ord("A") + offset.item()) for offset in torch.argmax(choice_probs, dim=-1)]

    def eval(self) -> Tuple[dict, Dict[str, dict]]:
        if "token" in inspect.signature(cached_file).parameters:
            kwargs = {"token": self.model_args.hf_hub_token}
        elif "use_auth_token" in inspect.signature(cached_file).parameters: # for transformers==4.31.0
            kwargs = {"use_auth_token": self.model_args.hf_hub_token}

        mapping = cached_file(
            path_or_repo_id = os.path.join(self.eval_args.task_dir, self.eval_args.task),
            filename="mapping.json",
            cache_dir=self.model_args.cache_dir,
            **kwargs
        )

        with open(mapping, "r", encoding="utf-8") as f:
            categorys: Dict[str, Dict[str, str]] = json.load(f)

        category_corrects = {subj: np.array([], dtype="bool") for subj in self.SUBJECTS}
        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        results = {}
        for subject in pbar:
            dataset = load_dataset(
                path=os.path.join(self.eval_args.task_dir, self.eval_args.task),
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token
            )
            pbar.set_postfix_str(categorys[subject]["name"])
            inputs, outputs, labels = [], [], []
            for i in trange(len(dataset[self.data_args.split]), desc="Formatting batches", position=1, leave=False):
                support_set = dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                query, resp, history = self.eval_template.format_example(
                    target_data=dataset[self.data_args.split][i],
                    support_set=support_set,
                    subject_name=categorys[subject]["name"],
                    use_history=self.template.use_history
                )
                input_ids, _ = self.template.encode_oneturn(
                    tokenizer=self.tokenizer, query=query, resp=resp, history=history
                )
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                labels.append(resp)

            for i in trange(0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False):
                batch_input = self.tokenizer.pad(
                    inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                ).to(self.model.device)
                preds = self.batch_inference(batch_input)
                outputs += preds

            corrects = (np.array(outputs) == np.array(labels))
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}

        pbar.close()
        #self._save_results(category_corrects, results)
        return category_corrects, results

    def _save_results(self, category_corrects: Dict[str, np.ndarray], results: Dict[str, Dict[int, str]]) -> List[str]:
        score_info = "\n".join([
            "{:>15}: {:.2f}".format(category_name, 100 * np.mean(category_correct))
            for category_name, category_correct in category_corrects.items() if len(category_correct)
        ])

        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=False)
            with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=2)

            with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)

if __name__ == "__main__":
    evaluator = AdvancedEvaluator()
    evaluator.eval()
