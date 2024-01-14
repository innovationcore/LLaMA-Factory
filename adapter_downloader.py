import argparse
import os

import boto3
import botocore


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM Factory Agent')

    # general args
    parser.add_argument('--project_name', type=str, default='llm_factory_trainer', help='name of project')
    parser.add_argument('--aws_endpoint_url', type=str, default='http://10.33.31.21:9000', help='name of project')
    parser.add_argument('--aws_access_key_id', type=str, default='hQYiBAhIGNP5xIIU79yO', help='location of dataset')
    parser.add_argument('--aws_secret_access_key', type=str, default='jWNVPYT6zkxamILIG4YYIXGUQZkeJC39wJO2yQRb', help='location of dataset')

    # get args
    args = parser.parse_args()

    BUCKET_NAME = 'my-bucket'  # replace with your bucket name
    KEY = 'my_image_in_s3.jpg'  # replace with your object key

    s3 = boto3.resource('s3',
        endpoint_url=args.aws_endpoint_url,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        aws_session_token=None)


    try:
        #s3.Bucket(BUCKET_NAME).download_file(KEY, 'my_local_image.jpg')

        my_bucket = s3.Bucket('llmadapters')

        for my_bucket_object in my_bucket.objects.all():
            print(my_bucket_object.key)


    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
