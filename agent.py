import argparse
import asyncio
import re

def extract_string_between_curly_braces(text):

  text = text.decode("utf-8")
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

def stdout_callback(x):
    print('stdout', x)
    json_str = extract_string_between_curly_braces(x)


def stderror_callback(x):
    print('stderror', x)

def main():


    #84%|████████▍ | 443/525 [1:58:2{'loss': 0.1842, 'learning_rate': 3.326959847036329e-05, 'epoch': 2.5}
    print(args.project_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM Factory Agent')

    # general args
    parser.add_argument('--project_name', type=str, default='llm_factory_trainer', help='name of project')
    parser.add_argument('--dataset_path', type=str, default='dataset.csv', help='location of dataset')

    # get args
    args = parser.parse_args()

    print(execute(
        ["bash", "-c", "ls -la"],
        lambda x: stdout_callback(x),
        lambda x: stderror_callback(x),
    ))