import os
import requests
import json
from clearml import Task
import subprocess
import time
from datetime import datetime

#for i in os.environ.items():
#    print(i)

while True:
    time.sleep(30) # 30 seconds between checking queue
    url = 'https://data.ai.uky.edu/llm-upload/api/get-queued-jobs'

    headers = {
            'apiKey':'a3495448-80b0-4342-9533-31f8eab76a0d'
    }

    data = None
    with requests.get(url, params=None, headers=headers) as response:

        if response.status_code != 200: # check if http request was successful. exit if not.
            print(f'Error: {response.status_code}')
            print(response.text)
            #exit(1)
            continue

        data = response.json()
    for i in data:
        print(i)
    if len(data) == 0:
        print('No jobs in queue')
        continue
    ids = []
    for key in data:
        ids.append(key)
    print(ids)
    #print(f'in queue: {ids}')
    #print(f'key: {ids[0]}')
    #print(json.dumps(data[keys[0]], indent=2))
    #exit(0)
    #new_args = data[keys[0]]
    for id in ids:
        print(f'|{datetime.now().strftime("%H:%M:%S")}| queuing {id}...')
        command = f'SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf /cm/shared/apps/slurm/23.02.5/bin/sbatch /project/ibi-staff/llmfactory/train_job {id}'
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"stdout: {result.stdout}", end='')
        print(f"stderr: {result.stderr}")
        success = False
        if result.returncode == 0:
            success = True
        if not success:
            print(f'Error submitting job {id}')
            continue
    ## On success ##
    url = 'https://data.ai.uky.edu/llm-upload/api/update-DGX-queue'
    headers = {
        'apiKey':'a3495448-80b0-4342-9533-31f8eab76a0d',
        'Content-Type': 'application/json'
    }
    body = {'ids':ids}
    json_body = json.dumps(body)
    print(f'posting ids: \n{json_body}')
    with requests.post(url, data=json_body, headers=headers) as response:
        print(response.status_code)
        print(response.text)
        print('\n\n')
