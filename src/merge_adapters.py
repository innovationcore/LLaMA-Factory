import json
import os
import gc
import os.path
import random
from statistics import mean
from time import sleep

import torch
from peft import PeftModel, PeftConfig
import numpy as np

def get_log(log_path):

    with open(log_path) as f:
        optuna_journal = json.load(f)

    print(optuna_journal)

def main():

    log_path = 'optuna-journal.log'
    get_log(log_path)

if __name__ == "__main__":
    main()
