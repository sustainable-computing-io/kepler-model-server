## get client
# client = new_<provider>_client(args) 
## upload all files in mnt path 
# <provider>_upload(client, mnt_path) 

model_dir="models"
data_dir="data"
machine_spec_dir="machine_spec"

import os

def aws_list_keys(client, bucket_name, prefix):
    response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    return [obj['Key'] for obj in response.get('Contents', [])]

def ibmcloud_list_keys(client, bucket_name, prefix):
    bucket_obj = client.Bucket(bucket_name)
    data_response = bucket_obj.objects.filter(Prefix=prefix)
    return [obj.key for obj in data_response]

def get_bucket_file_map(client, bucket_name, machine_id, mnt_path, pipeline_name, list_func):
    bucket_file_map = dict()
    top_key_path = ""
    if machine_id is not None and machine_id != "":
        top_key_path = "/" + machine_id
    # add data key map
    data_path = os.path.join(mnt_path, data_dir)
    datapath_prefix = top_key_path + "/data"
    keys = list_func(client, bucket_name, datapath_prefix)
    for key in keys:
        filepath = key.replace(datapath_prefix, data_path)
        bucket_file_map[key] = filepath
    # add model key map
    model_path = os.path.join(mnt_path, model_dir, pipeline_name)
    model_predix = "/models/" + pipeline_name
    keys = list_func(client, bucket_name, model_predix)
    for key in keys:
        filepath = key.replace(model_predix, model_path)
        bucket_file_map[key] = filepath
    return bucket_file_map

def aws_download(client, bucket_name, machine_id, mnt_path, pipeline_name):
    print("AWS Download")
    bucket_file_map = get_bucket_file_map(client, bucket_name, machine_id=machine_id, mnt_path=mnt_path, pipeline_name=pipeline_name, list_func=aws_list_keys)
    for key, filepath in bucket_file_map.items():
        print(key, filepath)
        dir = os.path.dirname(filepath)
        if not os.path.exists(dir):
            os.makedirs(dir)
        client.download_file(bucket_name, key, filepath)
        
def ibm_download(client, bucket_name, machine_id, mnt_path, pipeline_name):
    print("IBM Download")
    bucket_file_map = get_bucket_file_map(client, bucket_name, machine_id=machine_id, mnt_path=mnt_path, pipeline_name=pipeline_name, list_func=ibmcloud_list_keys)
    for key, filepath in bucket_file_map.items():
        print(key, filepath)
        dir = os.path.dirname(filepath)
        if not os.path.exists(dir):
            os.makedirs(dir)
        client.Bucket(bucket_name).download_file(key, filepath)
        
def add_common_args(subparser):
    subparser.add_argument("--bucket-name", help="Bucket name", required=True)
    subparser.add_argument("--mnt-path", help="Mount path", required=True)
    subparser.add_argument("--pipeline-name", help="Pipeline name")
    subparser.add_argument("--machine-id", help="Machine ID")

import argparse
import util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S3 Pusher")
    args = util.get_command(parser, add_common_args, ibm_download, aws_download)
    if hasattr(args, "new_client_func") and hasattr(args, "func"):
        client = args.new_client_func(args)
        args.func(client, args.bucket_name, args.machine_id, args.mnt_path, args.pipeline_name)
    else:
        parser.print_help()