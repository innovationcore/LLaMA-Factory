import argparse
import json
import os

import numpy as np


def gen_train_output():

    # 84%|████████▍ | 443/525 [1:58:2{'loss': 0.1842, 'learning_rate': 3.326959847036329e-05, 'epoch': 2.5}
    junk_text = '84%|████████▍ | 443/525 [1:58:2'
    learning_rate = 3.326959847036329e-05
    starting_loss = 3.0
    # np.arange() with step parameter
    for i in np.arange(0.0, 3.0, 0.1):
        epoch = round(i,2)
        loss = round(starting_loss - epoch,2)
        train_output = dict()
        train_output['loss'] = loss
        train_output['learning_rate'] = (learning_rate * epoch)
        train_output['epoch'] = epoch
        train_output = str(train_output)
        output = junk_text + train_output
        print(output)

    final_output = """
    wandb: Run summary:
    wandb:                    train/epoch 2.95
    wandb:              train/global_step 96
    wandb:            train/learning_rate 0.0
    wandb:                     train/loss 2.0302
    wandb:               train/total_flos 376535487873024.0
    wandb:               train/train_loss 2.1685
    wandb:            train/train_runtime 1665.7425
    wandb: train/train_samples_per_second 44.472
    wandb:   train/train_steps_per_second 0.058
    """
    print(final_output)
    figure_line = 'Figure saved: /workspace/outputmodels/medqa-textbooks-dataset_S-pt_R-64_A-16_E-3_LR-2e-4_M-TinyLlama-16x1.1B-Chat-v1.0-all/training_loss.png'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM Factory Dummy Trainer')

    # general args
    #parser.add_argument('--project_name', type=str, default='llm_factory_trainer', help='name of project')
    #parser.add_argument('--dataset_path', type=str, default='dataset.csv', help='location of dataset')

    # get args
    args = parser.parse_args()

    print('Print ENV:')
    for name, value in os.environ.items():
        print("{0}: {1}".format(name, value))

    gen_train_output()


