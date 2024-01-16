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
    print('stdout', x, end="")

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
            #some output can't be converted to json, just None it
            None
            #traceback.print_exc()
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
    print('stderror', x, end="")

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

def get_dataset_path():

    dataset_path = None
    # JSON file
    f = open(os.path.join(args.dataset_path,'dataset_info.json'), "r")

    # Reading from file
    dataset_info = json.loads(f.read())
    if args.dataset in dataset_info:
        dataset_path = os.path.join('data', dataset_info[args.dataset]['file_name'])

    return dataset_path

def prepare_dataset():

    #disable cache
    new_cache_limit = StorageManager.set_cache_file_limit(cache_file_limit=1)

    is_prepaired = False

    #args.dataset -> generic_instruct
    #args.dataset_path -> data
    #args.dataset_name -> custom_dataset
    #args.dataset_project -> datasets
    #args.dataset_file -> example_generic_instruct.json

    #download the dataset
    temp_download_dir = os.path.join(args.dataset_path, str(uuid.uuid4()))

    local_dataset = Dataset.get(
        dataset_name=args.dataset_name, dataset_project=args.dataset_project
    ).get_mutable_local_copy(temp_download_dir)

    print('Downloaded', args.dataset_name, 'to', temp_download_dir)

    #prepare custom dataset location

    custom_dataset_path = get_dataset_path()
    print('custom_dataset_path:', custom_dataset_path)
    #adjust for fixed path of container
    custom_dataset_path = os.path.join(args.training_root, custom_dataset_path)
    print('fixed custom_dataset_path:', custom_dataset_path)

    #/workspace/data/custom_data/generic_instruct.json

    #custom_dataset_filename = os.path.basename(custom_dataset_path)
    custom_dataset_dir = os.path.dirname(custom_dataset_path)
    print('custom_dataset_dir:', custom_dataset_dir)

    if os.path.exists(custom_dataset_dir):
        shutil.rmtree(custom_dataset_dir)
    os.makedirs(custom_dataset_dir)

    tmp_custom_dataset_path = os.path.join(temp_download_dir, args.dataset_name, args.dataset_name, args.dataset_file)
    print('tmp_custom_dataset_path:', tmp_custom_dataset_path)
    #tmp_custom_dataset_path = os.path.join(temp_download_dir, args.dataset_name, args.dataset_name, args.dataset_file)
    tmp_custom_dataset_path = os.path.join(temp_download_dir, args.dataset_name, args.dataset_file)
    #tmp_custom_dataset_path = os.path.join(args.training_root, tmp_custom_dataset_path)
    #print('fixed tmp_custom_dataset_path:', tmp_custom_dataset_path)
    if os.path.exists(tmp_custom_dataset_path):
        #print(custom_dataset_path)
        #print(tmp_custom_dataset_path)
        shutil.move(tmp_custom_dataset_path, custom_dataset_path)
        is_prepaired = True
        print('tmp_custom_dataset_path:', tmp_custom_dataset_path, 'moved to custom_dataset_path:', custom_dataset_path)
    else:
        print('Error: tmp_custom_dataset_path:' ,tmp_custom_dataset_path,'does not exist!')

    #clean up tmp dir
    print('removing tmp_custom_dataset_path:', tmp_custom_dataset_path)
    shutil.rmtree(temp_download_dir)
    print('remove clearml storage cache:')

    return is_prepaired

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
    #parser.add_argument('--dataset_sha1', type=str, default='bb2844e2293e4aa78acf05e9a7705c8d8deb0d62', help='location of dataset')
    parser.add_argument('--output_model', type=str, default='custom_adapter', help='location of dataset')

    #dataset params
    parser.add_argument('--dataset_project', type=str, default='datasets', help='location of dataset')
    parser.add_argument('--dataset_name', type=str, default='example_custom_dataset', help='location of dataset')
    parser.add_argument('--dataset_file', type=str, default='example_generic_instruct.json', help='location of dataset')
    parser.add_argument('--training_root', type=str, default='/workspace', help='location of dataset')

    # get args
    args = parser.parse_args()

    training_cmd = 'clearml_train.sh'
    training_cmd = os.path.join(args.training_root, training_cmd)

    print('Starting ClearML Task')

    task = Task.init(project_name=args.project_name, task_name=args.task_name)

    print('pre-dataset')
    is_dataset_prepared = prepare_dataset()
    print('post-dataset')

    if is_dataset_prepared:

        Logger.current_logger().report_text("Dataset prepared, starting training.", print_console=True)

        # set env vars for run
        set_env()

        execute(
            ["bash", "-c", training_cmd],
            lambda x: stdout_callback(x),
            lambda x: stderror_callback(x)
        )

        Logger.current_logger().report_text("Uploading adapter.", print_console=True)

        # at this point might as well upload zip, we will want to run directly from S3 at some point
        task.upload_artifact('adapter', artifact_object=os.path.join('/workspace/outputmodels/custom_adapter'))

        # adapter_config.json
        # "base_model_name_or_path": "/data/llama-2-7b-chat-hf",

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
        #dataset_validation_error = 'dataset: ' + args.dataset + ' dataset_sha1:' + args.dataset_sha1 + " != local_dataset_sha1:" + dataset_sha1
        Logger.current_logger().report_text('Was unable to prepare datasets.', print_console=True)


