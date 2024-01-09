import argparse
import asyncio
import json
import re

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
    print('upload_training_stats: UPLOAD TRAINING STATS:', training_stats)

def update_training_metrics(metric_key, metric_value):
    print('update_training_metrics: UPLOAD TRAINING METRIC:','metric_key:', metric_key, 'metric_value:', metric_value)

def stdout_callback(x):

    # 84%|████████▍ | 443/525 [1:58:2{'loss': 0.1842, 'learning_rate': 3.326959847036329e-05, 'epoch': 2.5}
    x = x.decode("utf-8")

    training_stats_keys = ['loss','learning_rate','epoch']

    json_str = extract_string_between_curly_braces(x)
    if json_str is not None:
        json_str = '{' + json_str + '}'
        json_str = json_str.replace('\'','"')
        training_stats = json.loads(json_str)
        if training_stats_keys == list(training_stats.keys()):
            upload_training_stats(training_stats)

    wandb_keys = ['train/epoch', 'train/global_step', 'train/learning_rate', 'train/loss',
                  'train/total_flos', 'train/train_loss', 'train/train_runtime',
                  'train/train_samples_per_second', 'train/train_steps_per_second']

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
    print('stderror', x)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM Factory Agent')

    # general args
    parser.add_argument('--project_name', type=str, default='llm_factory_trainer', help='name of project')
    parser.add_argument('--dataset_path', type=str, default='dataset.csv', help='location of dataset')

    # get args
    args = parser.parse_args()

    training_cmd = 'python3 dummy_train.py'

    execute(
        ["bash", "-c", training_cmd],
        lambda x: stdout_callback(x),
        lambda x: stderror_callback(x)
    )