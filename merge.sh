#!/bin/bash

python3 src/merge_adapters.py --journal_path=optuna-journal.log --basemodel=basemodels/llama-2-7b-hf/ --adapters_path=models/adapters/ --export_path=models/junk