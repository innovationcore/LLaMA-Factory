import argparse
import json
import os
import tempfile

import boto3
import botocore
import zipfile

def get_adapter_info():

    adapter_info = None
    adapter_info_path = os.path.join(args.adapter_repo_path, args.adapter_info)

    if os.path.exists(adapter_info_path):
        f = open(adapter_info_path, "r")
        adapter_info = json.loads(f.read())

    if adapter_info == None:
        adapter_info = dict()

    return adapter_info

def set_adapter_info(adapter_info):

    adapter_info_path = os.path.join(args.adapter_repo_path, args.adapter_info)

    json_object = json.dumps(adapter_info, indent=4)
    with open(adapter_info_path, "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM Factory Agent')

    # general args
    parser.add_argument('--project_name', type=str, default='llm_factory_trainer', help='name of project')
    # WARNING
    # I should not be putting this here, but it is a local debug S3, default needs to be removed
    parser.add_argument('--aws_endpoint_url', type=str, default='http://10.33.31.21:9000', help='name of project')
    parser.add_argument('--aws_access_key_id', type=str, default='hQYiBAhIGNP5xIIU79yO', help='location of dataset')
    parser.add_argument('--aws_secret_access_key', type=str, default='jWNVPYT6zkxamILIG4YYIXGUQZkeJC39wJO2yQRb', help='location of dataset')
    parser.add_argument('--adapter_repo_path', type=str, default='adapters', help='location of dataset')
    parser.add_argument('--adapter_info', type=str, default='adapter_info.json', help='location of dataset')
    # WARNING

    # get args
    args = parser.parse_args()

    #if repo does not exist create it
    isExist = os.path.exists(args.adapter_repo_path)
    if not isExist:
        os.makedirs(args.adapter_repo_path)

    #get adapter info
    adapter_info = get_adapter_info()

    s3 = boto3.resource('s3',
        endpoint_url=args.aws_endpoint_url,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        aws_session_token=None)

    bucket = 'llmadapters'
    prefix = "llm_factory_trainer/"


    try:

        my_bucket = s3.Bucket(bucket)
        prefix = 'llm_factory_trainer/'

        for object in my_bucket.objects.filter(Prefix=prefix):
            if 'custom_adapter.zip' in object.key:
                job_id = object.key.split('/')[1].split('.')[0]

                if job_id not in adapter_info:

                    print('New adapter for job_id:', job_id, 'found.')
                    tmp_adapter_save_path = os.path.join(tempfile.gettempdir(), job_id + '.zip')
                    print('Downloading: ', tmp_adapter_save_path)
                    my_bucket.download_file(object.key, tmp_adapter_save_path)
                    print('Saving adapter to repo')
                    adapter_save_path = os.path.join(args.adapter_repo_path, job_id)
                    with zipfile.ZipFile(tmp_adapter_save_path, 'r') as zip_ref:
                        zip_ref.extractall(adapter_save_path)
                    print('Removing temporary files:', tmp_adapter_save_path)
                    #remove temp file
                    os.remove(tmp_adapter_save_path)

                    #record
                    adapter_info[job_id] = adapter_save_path

        set_adapter_info(adapter_info)

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
