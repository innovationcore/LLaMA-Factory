import argparse
import os
import tempfile

import boto3
import botocore
import zipfile


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM Factory Agent')

    # general args
    parser.add_argument('--project_name', type=str, default='llm_factory_trainer', help='name of project')
    parser.add_argument('--aws_endpoint_url', type=str, default='http://10.33.31.21:9000', help='name of project')
    parser.add_argument('--aws_access_key_id', type=str, default='hQYiBAhIGNP5xIIU79yO', help='location of dataset')
    parser.add_argument('--aws_secret_access_key', type=str, default='jWNVPYT6zkxamILIG4YYIXGUQZkeJC39wJO2yQRb', help='location of dataset')
    parser.add_argument('--adapter_repo_path', type=str, default='adapters', help='location of dataset')

    # get args
    args = parser.parse_args()

    #if repo does not exist create it
    isExist = os.path.exists(args.adapter_repo_path)
    if not isExist:
        os.makedirs(args.adapter_repo_path)

    s3 = boto3.resource('s3',
        endpoint_url=args.aws_endpoint_url,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        aws_session_token=None)

    bucket = 'llmadapters'
    prefix = "llm_factory_trainer/"

    try:

        my_bucket = s3.Bucket('llmadapters')
        prefix = 'llm_factory_trainer/'

        for object in my_bucket.objects.filter(Prefix=prefix):
            if 'custom_adapter.zip' in object.key:
                job_id = object.key.split('/')[1]

                tmp_adapter_save_path = os.path.join(tempfile.gettempdir(), job_id + '.zip')
                my_bucket.download_file(object.key, tmp_adapter_save_path)

                adapter_save_path = os.path.join(args.adapter_repo_path, job_id)

                with zipfile.ZipFile(tmp_adapter_save_path, 'r') as zip_ref:
                    zip_ref.extractall(adapter_save_path)

                #remove temp file
                os.remove(tmp_adapter_save_path)


    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
