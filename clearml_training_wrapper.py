import argparse
import asyncio
import json
import os
import re
import shutil
import hashlib
from glob import glob
from os.path import isfile, join

import pandas as pd
from clearml import Task, Dataset
from clearml import Logger
import yaml
from pathlib import Path

from clearml.utilities import pyhocon
import boto3
from s3transfer import TransferConfig, S3Transfer
step_count = 1

user_home = str(Path.home())

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

def clean_custom_adapter(custom_adapter_save_path):

    pattern = os.path.join(custom_adapter_save_path, "checkpoint-*")

    for item in glob(pattern):
        if not os.path.isdir(item):
            continue
        shutil.rmtree(item)

    shutil.rmtree(os.path.join(custom_adapter_save_path, "runs"))


def create_custom_dataset_config(custom_task_data_path, dataset_cache_path):

    custom_dataset_config_path = None

    dataset_config_template_path = '/app/config/dataset_info.json'

    with open(dataset_config_template_path) as f:
        dataset_info = json.load(f)

    if args.dataset in dataset_info:
        custom_dataset_config = dict()
        custom_dataset_config[args.dataset] = dataset_info[args.dataset]
        dataset_path = os.path.join(dataset_cache_path, args.dataset_file, args.dataset_file)
        custom_dataset_config[args.dataset]['file_name'] = dataset_path
        if os.path.exists(dataset_path):
            if os.path.exists(custom_task_data_path):
                shutil.rmtree(custom_task_data_path)
            os.makedirs(custom_task_data_path)

            custom_dataset_config_path = os.path.join(custom_task_data_path, 'dataset_info.json')
            print('Saving custom dataset config:', custom_dataset_config_path)
            with open(custom_dataset_config_path, 'w', encoding='utf-8') as f:
                json.dump(custom_dataset_config, f, ensure_ascii=False, indent=4)

            if not os.path.exists(custom_dataset_config_path):
                custom_dataset_config_path = None

        else:
            print('dataset not found:', dataset_path)

    return custom_dataset_config_path

def create_training_params(custom_task_data_path, adapter_save_path):

    deepseed_config = ''
    if args.ds_stage is not None:
        deepseed_config = '/app/config/ds_z' + str(args.ds_stage) + '_config.json'

    flash_attn_config = ''
    if args.flash_attn_version is not None:
        flash_attn_config = 'fa' + str(args.flash_attn_version)

    training_params = {

        "model_name_or_path": args.model,

        "stage": args.stage,
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_target": args.lora_target,
        "deepspeed": deepseed_config,
        "flash_attn": flash_attn_config,

        "dataset_dir": custom_task_data_path,
        "dataset": args.dataset,
        "template": args.template,
        "cutoff_len": 8096,
        "max_samples": 1000000000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,

        "output_dir": adapter_save_path,
        "logging_steps": 10,
        "save_strategy": "no",
        "plot_loss": True,
        "overwrite_output_dir": True,

        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "num_train_epochs": args.epoch,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
    }

    if len(training_params['deepspeed']) == 0:
        del training_params['deepspeed']

    if len(training_params['flash_attn']) == 0:
        del training_params['flash_attn']

    training_params_path = os.path.join(custom_task_data_path,'training_params.yaml')
    with open(training_params_path, 'w') as f:
        yaml.dump(training_params, f, default_flow_style=False)

    return training_params_path

def prepare_dataset(task_id):

    custom_task_data_path = os.path.join(args.dataset_path, task_id)

    train_dataset = Dataset.get(dataset_name=args.dataset_name, dataset_project=args.dataset_project)
    dataset_cache_path = train_dataset.get_local_copy()

    print('Creating custom dataset config for:', dataset_cache_path)
    custom_dataset_config_path = create_custom_dataset_config(custom_task_data_path, dataset_cache_path)

    print('custom_dataset_config_path:', custom_dataset_config_path)
    #custom_dataset_path: /app/custom_data/generic_instruct.json

    custom_adapter_save_path = os.path.join(custom_task_data_path, 'adapter')
    training_params_path = create_training_params(custom_task_data_path, custom_adapter_save_path)
    print('training_params_path:', training_params_path)

    return custom_dataset_config_path, training_params_path, custom_task_data_path, dataset_cache_path, custom_adapter_save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Factory Agent')

    # General arguments
    parser.add_argument('--project_name', type=str, default='llm_factory_trainer', help='name of project')
    parser.add_argument('--task_name', type=str, default='trainer_template_v0', help='name of project')
    parser.add_argument('--ds_stage', type=int, help='location of dataset')
    parser.add_argument('--lora_rank', type=int, default=8, help='location of dataset')
    parser.add_argument('--lora_alpha', type=int, default=8, help='location of dataset')
    parser.add_argument('--lora_target', type=str, default='all', help='location of dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='location of dataset')
    parser.add_argument('--flash_attn_version', type=int, help='location of dataset')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='location of dataset')
    parser.add_argument('--epoch', type=float, default=1.0, help='location of dataset')
    parser.add_argument('--lr', type=float, default=1e-4, help='location of dataset')
    parser.add_argument('--template', type=str, default='llama3', help='location of dataset')
    parser.add_argument('--model', type=str, default='/app/basemodels/Meta-Llama-3-8B-Instruct', help='location of dataset')
    parser.add_argument('--stage', type=str, default='sft', help='location of dataset')
    parser.add_argument('--dataset_path', type=str, default='/app/custom_data', help='location of dataset')
    parser.add_argument('--dataset', type=str, default='generic_instruct', help='location of dataset')
    #parser.add_argument('--output_model', type=str, default='/app/custom_adapter', help='location of dataset')

    # Dataset parameters
    parser.add_argument('--dataset_project', type=str, default='datasets', help='location of dataset')
    parser.add_argument('--dataset_name', type=str, default='example_generic_instruct', help='location of dataset')
    parser.add_argument('--dataset_file', type=str, default='example_generic_instruct.json', help='location of dataset')
    #parser.add_argument('--clearml_cache', type=str, default=os.path.join(user_home,'.clearml/cache'), help='location of dataset')
    #parser.add_argument('--clearml_cache', type=str, default='/app/cache', help='location of dataset')
    parser.add_argument('--job_id', type=str, help='id of job')

    args = parser.parse_args()

    '''
    tmp_task_id = str(uuid.uuid4())
    tmp_custom_task_data_path = os.path.join('/app/cache', tmp_task_id)
    os.mkdir(tmp_custom_task_data_path)

    os.environ["HOME"] = tmp_custom_task_data_path
    os.environ["OLDPW"] = tmp_custom_task_data_path

    os.environ["CLEARML_VENVS_BUILDS"] = os.path.join(tmp_custom_task_data_path, '.clearml','venvs-builds')
    os.environ["CLEARML_VCS_CACHE"] = os.path.join(tmp_custom_task_data_path, '.clearml', 'vcs-cache')
    os.environ["CLEARML_PIP_CACHE"] = os.path.join(tmp_custom_task_data_path, '.clearml', 'pip-download-cache')
    os.environ["CLEARML_DOCKER_PIP_CACHE"] = os.path.join(tmp_custom_task_data_path, '.clearml', 'pip-cache')
    os.environ["CLEARML_APT_CACHE"] = os.path.join(tmp_custom_task_data_path, '.clearml', 'apt-cache')

    if "PYTHONPATH" in os.environ:
        print('PYTHONPATH 0:', os.environ["PYTHONPATH"])

    print('Setting HOME as:', os.environ["HOME"])
    #os.environ["CLEARML_LOG_ENVIRONMENT"] = tmp_custom_task_data_path
    os.environ["CLEARML_TASK_NO_REUSE"] = '1'
    os.environ["CLEARML_LOG_LEVEL"] = 'INFO'
    '''

    print('ENVS:')
    for name, value in os.environ.items():
        print("{0}: {1}".format(name, value))

    control_node = False

    if "PMIX_RANK" in os.environ:
        if os.environ["PMIX_RANK"] == '0':
            control_node = True
    else:
        control_node = True

    print('Starting ClearML Task')
    task = Task.init(project_name=args.project_name, task_name=args.task_name)
    task_id = str(task.current_task().id)
    print('Task_id:', task_id)

    '''
    if "PYTHONPATH" in os.environ:
        print('PYTHONPATH 1:', os.environ["PYTHONPATH"])
        #del os.environ['PYTHONPATH']
    if "PYTHONPATH" in os.environ:
        print('PYTHONPATH 2:', os.environ["PYTHONPATH"])
    '''

    custom_dataset_config_path, training_params_path, custom_task_data_path, dataset_cache_path, custom_adapter_save_path = prepare_dataset(task_id)

    if custom_dataset_config_path is not None:
        print('Dataset is prepared successfully. Starting training...')

        #allow script to cleanup on failure
        os.environ["CLEARML_CUSTOM_TASK_DATA_PATH"] = custom_task_data_path

        # Setting environment variables
        set_env()

        # Call llamafactory-cli with the training parameters file
        command = ['llamafactory-cli', 'train', training_params_path]
        rc = execute(command, stdout_callback, stderror_callback)

        if rc != 0:
            raise ValueError(f"Training failed with return code {rc}")

        if (args.job_id is not None) and (control_node):

            # remove checkpoints
            print('Cleaning adapter before upload')
            clean_custom_adapter(custom_adapter_save_path)

            Logger.current_logger().report_text("Uploading adapter.", print_console=True)

            s3_bucket = 'llmadapters'
            #os.environ['CLEARML_CONFIG_FILE']
            #clearml_config_path = os.path.join(os.path.expanduser('~'), 'clearml.conf')
            clearml_config_path = os.environ['CLEARML_CONFIG_FILE']
            config = pyhocon.ConfigFactory.parse_file(clearml_config_path)
            for record in config['sdk']['aws']['s3']['credentials']:
                if record['bucket'] == s3_bucket:
                    s3_endpoint = 'http://' + record['host']
                    s3_key = record['key']
                    s3_secret = record['secret']

            myconfig = TransferConfig(

                multipart_threshold=9999999999999999,  # workaround for 'disable' auto multipart upload
                # multipart_threshold=1,  # workaround for 'disable' auto multipart upload
                max_concurrency=10,
                num_download_attempts=10,
            )

            s3_client = boto3.client('s3',
                                     endpoint_url=s3_endpoint,
                                     aws_access_key_id=s3_key,
                                     aws_secret_access_key=s3_secret,
                                     aws_session_token=None)

            transfer = S3Transfer(s3_client, myconfig)

            print('Uploading adapter: ', custom_adapter_save_path)
            dataset_files = [f for f in os.listdir(custom_adapter_save_path) if isfile(join(custom_adapter_save_path, f))]
            for dataset_file in dataset_files:
                local_dataset_path = os.path.join(custom_adapter_save_path, dataset_file)
                remote_dataset_path = args.job_id + '/' + dataset_file
                print('Remote dataset path:', remote_dataset_path)
                response = transfer.upload_file(local_dataset_path, s3_bucket, remote_dataset_path)


    else:
        raise Exception('Dataset preparation failed!')

    task.close()

    #remove temp
    #shutil.rmtree(tmp_custom_task_data_path)
    if control_node:
        print('Finished Training, cleaning files')
        clean_paths = [custom_task_data_path, dataset_cache_path]
        for path in clean_paths:
            if path is not None:
                if os.path.exists(path):
                    print('Removing path:', path)
                    shutil.rmtree(path)

    #for name, value in os.environ.items():
    #    print("{0}: {1}".format(name, value))