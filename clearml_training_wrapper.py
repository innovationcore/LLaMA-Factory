import argparse
import asyncio
import json
import os
import re
import shutil
import traceback
import hashlib
import uuid
from os import listdir
from os.path import isfile, join
import pandas as pd
from clearml import Task, Dataset, StorageManager
from clearml import Logger
import yaml

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
            limit=1024 * 1024 * 10,  # 10M buffer
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)

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
    print('update_training_metrics: UPLOAD TRAINING METRIC:', 'metric_key:', metric_key, 'metric_value:', metric_value)
    Logger.current_logger().report_single_value(metric_key, metric_value)

def stdout_callback(x):
    x = x.decode("utf-8")
    print('stdout', x, end="")

    training_stats_keys = ['loss', 'learning_rate', 'epoch']
    training_final_report_keys = ['train_runtime', 'train_samples_per_second', 'train_steps_per_second', 'train_loss', 'epoch']

    json_str = extract_string_between_curly_braces(x)
    if json_str is not None:
        json_str = '{' + json_str + '}'
        json_str = json_str.replace('\'', '"').lower()
        try:
            training_json_output = json.loads(json_str)
            if training_stats_keys == list(training_json_output.keys()):
                upload_training_stats(training_json_output)
            elif training_final_report_keys == list(training_json_output.keys()):
                for final_key, final_value in training_json_output.items():
                    update_training_metrics(final_key, final_value)
        except:
            None

    wandb_keys = ['train/epoch', 'train/global_step', 'train/learning_rate', 'train/loss',
                  'train/total_flos', 'train/train_loss', 'train/train_runtime',
                  'train/train_samples_per_second', 'train/train_steps_per_second']

    if 'wandb:' in x:
        for metric_key in wandb_keys:
            if metric_key in x:
                metric_value = x.split(metric_key)[1].strip()
                update_training_metrics(metric_key, metric_value)

def stderror_callback(x):
    x = x.decode("utf-8")
    print('stderror', x, end="")

def set_env():
    env_keys = []
    env_values = []

    for arg in vars(args):
        env_key = arg.upper()
        print(f'arg: {arg}')
        print(f'env_key: {env_key}')
        env_value = str(getattr(args, arg))
        print(f'env_value: {env_value}')
        os.environ[env_key] = env_value
        env_keys.append(env_key)
        env_values.append(env_value)

    data = {'env_keys': env_keys, 'env_values': env_values}
    df = pd.DataFrame.from_dict(data)
    Logger.current_logger().report_table(title='ENV Table', series='ENVs', iteration=0, table_plot=df)

def get_file_sha1(dataset_path):
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
    f = open('/app/custom_data/dataset_info.json', "r")
    dataset_info = json.loads(f.read())
    if args.dataset in dataset_info:
        dataset_path = os.path.join('data', dataset_info[args.dataset]['file_name'])
        dataset_sha1 = get_file_sha1(dataset_path)
    return dataset_sha1

def get_dataset_path():
    dataset_path = None
    f = open(os.path.join(args.dataset_path, 'dataset_info.json'), "r")
    dataset_info = json.loads(f.read())
    if args.dataset in dataset_info:
        dataset_path = os.path.join('data', dataset_info[args.dataset]['file_name'])
    return dataset_path

def get_custom_dataset_path():

    save_dataset_path = None

    dataset_info_path = os.path.join(args.dataset_path, 'dataset_info.json')

    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    if args.dataset in dataset_info:
        save_dataset_path = os.path.join(args.dataset_path, dataset_info[args.dataset]['file_name'])

    return save_dataset_path

def prepare_dataset():

    is_prepaired = False

    custom_dataset_path = get_custom_dataset_path()

    if custom_dataset_path is not None:

        temp_download_dir = os.path.join(args.dataset_path, str(uuid.uuid4()))
        print(f'temp_data_id: {uuid.uuid4()}')

        Dataset.get(
            dataset_name=args.dataset_name, dataset_project=args.dataset_project
        ).get_mutable_local_copy(temp_download_dir)

        print('Downloaded', args.dataset_name, 'to', temp_download_dir)

        print('fixed custom_dataset_path:', custom_dataset_path)

        custom_dataset_dir = os.path.dirname(custom_dataset_path)
        print('custom_dataset_dir:', custom_dataset_dir)

        if os.path.exists(custom_dataset_dir):
            shutil.rmtree(custom_dataset_dir)
        os.makedirs(custom_dataset_dir)

        tmp_custom_dataset_path = os.path.join(temp_download_dir, args.dataset_name, args.dataset_file)

        if os.path.exists(tmp_custom_dataset_path):
            shutil.move(tmp_custom_dataset_path, custom_dataset_path)
            is_prepaired = True
            print('tmp_custom_dataset_path:', tmp_custom_dataset_path, 'moved to custom_dataset_path:',
                  custom_dataset_path)
        else:
            print('Error: tmp_custom_dataset_path:', tmp_custom_dataset_path, 'does not exist!')

        print('removing tmp_custom_dataset_path:', tmp_custom_dataset_path)
        shutil.rmtree(temp_download_dir)
        print('remove clearml storage cache:')

    else:
        print('Error: save_dataset_path:', custom_dataset_path, ' for dataset', args.dataset, ' does not exist!')

    return is_prepaired


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Factory Agent')

    # General arguments
    parser.add_argument('--project_name', type=str, default='llm_factory_trainer', help='name of project')
    parser.add_argument('--task_name', type=str, default='dgx_trainer_template_v0', help='name of project')
    parser.add_argument('--lora_rank', type=int, default=8, help='location of dataset')
    parser.add_argument('--lora_alpha', type=int, default=8, help='location of dataset')
    parser.add_argument('--lora_target', type=str, default='all', help='location of dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='location of dataset')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='location of dataset')
    parser.add_argument('--epoch', type=float, default=1.0, help='location of dataset')
    parser.add_argument('--lr', type=float, default=1e-4, help='location of dataset')
    parser.add_argument('--template', type=str, default='llama3', help='location of dataset')
    parser.add_argument('--model', type=str, default='/app/basemodels/Meta-Llama-3-8B', help='location of dataset')
    parser.add_argument('--stage', type=str, default='sft', help='location of dataset')
    parser.add_argument('--dataset_path', type=str, default='/app/custom_data', help='location of dataset')
    parser.add_argument('--dataset', type=str, default='generic_instruct', help='location of dataset')
    parser.add_argument('--output_model', type=str, default='custom_adapter', help='location of dataset')

    # Dataset parameters
    parser.add_argument('--dataset_project', type=str, default='datasets', help='location of dataset')
    parser.add_argument('--dataset_name', type=str, default='example_generic_instruct.json', help='location of dataset')
    parser.add_argument('--dataset_file', type=str, default='example_generic_instruct.json', help='location of dataset')

    args = parser.parse_args()

    training_params = {

        "model_name_or_path": args.model,

        "stage": args.stage,
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_target": args.lora_target,
        "deepspeed": "/app/ds_z3_config.json",

        "dataset_dir": args.dataset_path,
        "dataset": args.dataset,
        "template": args.template,
        "cutoff_len": 8096,
        "max_samples": 100000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,

        "output_dir": "/app/outputmodels/sft",
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,

        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "num_train_epochs": args.epoch,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "fp16": True,
        "ddp_timeout": 180000000,

        "val_size": 0.1,
        "per_device_eval_batch_size": 1,
        "eval_strategy": "steps",
        "eval_steps": 500
    }

    training_params_file = 'training_params.yaml'
    with open(training_params_file, 'w') as f:
        yaml.dump(training_params, f, default_flow_style=False)

    task = Task.init(project_name=args.project_name, task_name=args.task_name, output_uri=True)

    if prepare_dataset():
        print('Dataset is prepared successfully. Starting training...')
    else:
        raise Exception('Dataset preparation failed!')

    # Setting environment variables
    set_env()

    # Call llamafactory-cli with the training parameters file
    command = ['llamafactory-cli', 'train', training_params_file]
    rc = execute(command, stdout_callback, stderror_callback)

    if rc != 0:
        raise ValueError(f"Training failed with return code {rc}")

    task.close()
