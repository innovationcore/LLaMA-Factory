#run this to create clearml dataset

import argparse
import os
from clearml import StorageManager, Dataset
import boto3
import botocore
from os import listdir
from os.path import isfile, join

from clearml.storage.helper import StorageHelper

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM Factory Agent')

    # general args
    parser.add_argument('--project_name', type=str, default='llm_factory_trainer', help='name of project')
    #WARNING
    #I should not be putting this here, but it is a local debug S3, default needs to be removed
    #parser.add_argument('--aws_endpoint_url', type=str, default='http://localhost:9000', help='name of project')
    parser.add_argument('--aws_endpoint_url', type=str, default='http://10.10.5.76:9000', help='name of project')

    parser.add_argument('--aws_access_key_id', type=str, default='rHUYeAk58Ilhg6iUEFtr', help='location of dataset')
    parser.add_argument('--aws_secret_access_key', type=str, default='IVimdW7BIQLq9PLyVpXzZUq8zS4nLfrsoiZSJanu', help='location of dataset')
    #WARNING
    parser.add_argument('--local_dataset_path', type=str, default='data/example_custom_dataset', help='location of dataset')
    parser.add_argument('--bucket', type=str, default='datasets', help='location of dataset')
    parser.add_argument('--remote_dataset_path', type=str, default='example_custom_dataset', help='location of dataset')

    # get args
    args = parser.parse_args()

    s3 = boto3.client('s3',
        endpoint_url=args.aws_endpoint_url,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        aws_session_token=None)

    try:

        if os.path.exists(args.local_dataset_path):

            dataset_files = [f for f in listdir(args.local_dataset_path) if isfile(join(args.local_dataset_path, f))]
            for dataset_file in dataset_files:
                local_dataset_path = os.path.join(args.local_dataset_path, dataset_file)
                remote_dataset_path = args.remote_dataset_path + '/' + dataset_file
                response = s3.upload_file(local_dataset_path, args.bucket, remote_dataset_path)
                print(local_dataset_path, args.bucket, remote_dataset_path)


            #remote_url = 's3://10.33.31.21:9000/datasets/example_custom_dataset/'
            #remote_url = 's3://localhost:9000/datasets/example_custom_dataset/'
            remote_url = 's3://10.10.5.76:9000/datasets/example_custom_dataset/'
            helper = StorageHelper.get(remote_url)
            helper_list_result = helper.list(prefix=remote_url, with_metadata=True)
            #print(helper_list_result)
            #exit(0)

            # init clearml
            manager = StorageManager()
            # if all uploaded register with clearml
            dataset = Dataset.create(
                dataset_name=args.remote_dataset_path, dataset_project="datasets"
            )

            #add files to clearml
            dataset_path = args.remote_dataset_path
            source_url = args.aws_endpoint_url.replace('http://', 's3://') + '/' + args.bucket + '/' + dataset_path
            print('source_url: ', source_url, 'dataset_path:', dataset_path)
            print('add external')
            dataset.add_external_files(source_url=source_url, dataset_path=dataset_path, verbose=True)
            dataset.upload(verbose=True)
            dataset.finalize(verbose=True)


        else:
            print('Missing local_dataset_path=', args.local_dataset_path)

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
