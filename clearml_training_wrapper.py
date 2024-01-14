import argparse
import asyncio
import json
import os
import re
import traceback
import hashlib
from os import listdir
from os.path import isfile, join
import pandas as pd
from clearml import Task
from clearml import Logger

step_count = 1

def extract_string_between_curly_braces(text):

  match = re.search(r'\{(.*?)\}', text)
  if match:
    return match.group(1)
  else:
    return None

async def _read_stream(stream, cb):
    while True:
        line = await stream.readline()
        if line:
            cb(line)
        else:
            break

async def _stream_subprocess(cmd, stdout_cb, stderr_cb):
    process = await asyncio.create_subprocess_exec(*cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    await asyncio.gather(
        _read_stream(process.stdout, stdout_cb),
        _read_stream(process.stderr, stderr_cb)
    )
    return await process.wait()


def execute(cmd, stdout_cb, stderr_cb):
    loop = asyncio.get_event_loop()
    rc = loop.run_until_complete(
        _stream_subprocess(
            cmd,
            stdout_cb,
            stderr_cb,
    ))
    loop.close()
    return rc

def upload_training_stats(training_stats):

    global step_count  # declare a to be a global

    print('upload_training_stats:', training_stats)
    Logger.current_logger().report_scalar("STEP_EPOCH", "step_epoch", iteration=step_count, value=training_stats['epoch'])
    Logger.current_logger().report_scalar("LOSS", "loss", iteration=step_count, value=training_stats['loss'])
    Logger.current_logger().report_scalar("LR", "lr", iteration=step_count, value=training_stats['learning_rate'])

    step_count += 1

def update_training_metrics(metric_key, metric_value):
    print('update_training_metrics: UPLOAD TRAINING METRIC:','metric_key:', metric_key, 'metric_value:', metric_value)
    Logger.current_logger().report_single_value(metric_key, metric_value)

def stdout_callback(x):

    x = x.decode("utf-8")
    #suppressing for security reasons
    #print('stdout', x, end="")

    training_stats_keys = ['loss','learning_rate','epoch']
    training_final_report_keys = ['train_runtime','train_samples_per_second','train_steps_per_second','train_loss', 'epoch']

    json_str = extract_string_between_curly_braces(x)
    if json_str is not None:
        json_str = '{' + json_str + '}'
        json_str = json_str.replace('\'','"').lower()
        try:
            training_json_output = json.loads(json_str)
            #print('decoded json:', json_str)
            if training_stats_keys == list(training_json_output.keys()):
                upload_training_stats(training_json_output)
            elif training_final_report_keys == list(training_json_output.keys()):
                for final_key, final_value in training_json_output.items():
                    update_training_metrics(final_key, final_value)

        except:
            traceback.print_exc()
            #print('failed to decode:', json_str)

    wandb_keys = ['train/epoch', 'train/global_step', 'train/learning_rate', 'train/loss',
                  'train/total_flos', 'train/train_loss', 'train/train_runtime',
                  'train/train_samples_per_second', 'train/train_steps_per_second']

    #typically disabled
    if 'wandb:' in x:
        for metric_key in wandb_keys:
            if metric_key in x:
                metric_value = x.split(metric_key)[1].strip()
                update_training_metrics(metric_key, metric_value)
    '''
    wandb:                    train/epoch 2.95
    wandb:              train/global_step 96
    wandb:            train/learning_rate 0.0
    wandb:                     train/loss 2.0302
    wandb:               train/total_flos 376535487873024.0
    wandb:               train/train_loss 2.1685
    wandb:            train/train_runtime 1665.7425
    wandb: train/train_samples_per_second 44.472
    wandb:   train/train_steps_per_second 0.058
    '''

def stderror_callback(x):
    x = x.decode("utf-8")
    #supressing for security reasons
    #print('stderror', x, end="")

def set_env():

    env_keys = []
    env_values = []

    for arg in vars(args):
        env_key = arg.upper()
        env_value = str(getattr(args, arg))
        #set env based on all args
        os.environ[env_key] = env_value
        #update report DF
        env_keys.append(env_key)
        env_values.append(env_value)

    data = {'env_keys': env_keys, 'env_values': env_values}
    df = pd.DataFrame.from_dict(data)
    Logger.current_logger().report_table(title='ENV Table', series='ENVs', iteration=0, table_plot=df)

def get_file_sha1(dataset_path):

    # BUF_SIZE is totally arbitrary, change for your app!
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    sha1 = hashlib.sha1()

    with open(dataset_path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest()

def validate_dataset():

    dataset_sha1 = None
    # JSON file
    f = open('data/dataset_info.json', "r")

    # Reading from file
    dataset_info = json.loads(f.read())
    if args.dataset in dataset_info:
        dataset_path = os.path.join('data', dataset_info[args.dataset]['file_name'])
        dataset_sha1 = get_file_sha1(dataset_path)

    return dataset_sha1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM Factory Agent')

    # general args
    parser.add_argument('--project_name', type=str, default='llm_factory_trainer', help='name of project')
    parser.add_argument('--task_name', type=str, default='trainer_template_v0', help='name of project')
    parser.add_argument('--lora_rank', type=int, default=8, help='location of dataset')
    parser.add_argument('--lora_alpha', type=int, default=8, help='location of dataset')
    parser.add_argument('--lora_target', type=str, default='all', help='location of dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='location of dataset')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='location of dataset')
    parser.add_argument('--epoch', type=float, default=1.0, help='location of dataset')
    parser.add_argument('--lr', type=float, default=2e-4, help='location of dataset')
    parser.add_argument('--template', type=str, default='default', help='location of dataset')
    parser.add_argument('--model', type=str, default='llama-2-7b-chat-hf', help='location of dataset')
    parser.add_argument('--stage', type=str, default='sft', help='location of dataset')
    parser.add_argument('--dataset_path', type=str, default='data', help='location of dataset')
    parser.add_argument('--dataset', type=str, default='generic_instruct', help='location of dataset')
    parser.add_argument('--dataset_sha1', type=str, default='bb2844e2293e4aa78acf05e9a7705c8d8deb0d62', help='location of dataset')
    parser.add_argument('--output_model', type=str, default='custom_adapter', help='location of dataset')

    # get args
    args = parser.parse_args()

    training_cmd = '/workspace/clearml_train.sh'

    print('Starting ClearML Task')

    dataset_sha1 = validate_dataset()
    error_notice = 'dataset: ' + args.dataset + ' dataset_sha1:' + args.dataset_sha1 + " != local_dataset_sha1:" + dataset_sha1

    task = Task.init(project_name=args.project_name, task_name=args.task_name)

    #validate the dataset
    dataset_sha1 = validate_dataset()
    if dataset_sha1 is not None:
        if dataset_sha1 == args.dataset_sha1:

            dataset_validation_notice = 'dataset: ' + args.dataset + ' dataset_sha1:' + args.dataset_sha1 + " == local_dataset_sha1:" + dataset_sha1
            Logger.current_logger().report_text(dataset_validation_notice, print_console=True)

            #set env vars for run
            set_env()

            execute(
                ["bash", "-c", training_cmd],
                lambda x: stdout_callback(x),
                lambda x: stderror_callback(x)
            )

            #at this point might as well upload zip, we will want to run directly from S3 at some point
            task.upload_artifact('adapter', artifact_object=os.path.join('/workspace/outputmodels/custom_adapter'))

            #adapter_config.json
            #"base_model_name_or_path": "/data/llama-2-7b-chat-hf",

            '''
            adapter_path = '/workspace/outputmodels/custom_adapter'
            adapter_files = [f for f in listdir(adapter_path) if isfile(join(adapter_path, f))]
        
            for adapter_file in adapter_files:
                task.upload_artifact(
                    'adapter_test', artifact_object=os.path.join(adapter_path, adapter_file)
                )
            '''
        else:
            # check dataset
            dataset_validation_error = 'dataset: ' + args.dataset + ' dataset_sha1:' + args.dataset_sha1 + " != local_dataset_sha1:" + dataset_sha1
            Logger.current_logger().report_text(dataset_validation_error, print_console=True)


