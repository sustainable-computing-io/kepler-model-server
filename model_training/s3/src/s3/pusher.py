## get client
# client = new_<provider>_client(args)
## upload all files in mnt path
# <provider>_upload(client, mnt_path)
import os
import argparse
from . import util

model_dir = "models"
data_dir = "data"
machine_spec_dir = "machine_spec"


def get_bucket_file_map(machine_id, mnt_path, query_data, idle_data):
    model_path = os.path.join(mnt_path, model_dir)
    bucket_file_map = dict()
    top_key_path = ""
    if machine_id is not None and machine_id != "":
        top_key_path = "/" + machine_id
    if os.path.exists(model_path):
        for root, _, files in os.walk(model_path):
            for file in files:
                filepath = os.path.join(root, file)
                key = filepath.replace(model_path, "/models")
                bucket_file_map[key] = filepath
    data_path = os.path.join(mnt_path, data_dir)
    for data_filename in [query_data, idle_data]:
        if data_filename is not None:
            filepath = os.path.join(data_path, data_filename + ".json")
            if os.path.exists(filepath):
                key = filepath.replace(data_path, top_key_path + "/data")
                bucket_file_map[key] = filepath
    filepath = os.path.join(data_path, machine_spec_dir, machine_id + ".json")
    if os.path.exists(filepath):
        key = filepath.replace(data_path, top_key_path + "/data")
        bucket_file_map[key] = filepath
    return bucket_file_map


def aws_upload(client, bucket_name, machine_id, mnt_path, query_data, idle_data):
    print("AWS Upload")
    bucket_file_map = get_bucket_file_map(machine_id=machine_id, mnt_path=mnt_path, query_data=query_data, idle_data=idle_data)
    for key, filepath in bucket_file_map.items():
        print(key, filepath)
        client.upload_file(filepath, bucket_name, key)


def ibm_upload(client, bucket_name, machine_id, mnt_path, query_data, idle_data):
    print("IBM Upload")
    bucket_file_map = get_bucket_file_map(machine_id=machine_id, mnt_path=mnt_path, query_data=query_data, idle_data=idle_data)
    for key, filepath in bucket_file_map.items():
        print(key, filepath)
        client.Object(bucket_name, key).upload_file(filepath)


def add_common_args(subparser):
    subparser.add_argument("--bucket-name", help="Bucket name", required=True)
    subparser.add_argument("--mnt-path", help="Mount path", required=True)
    subparser.add_argument("--query-data", help="Query data filename")
    subparser.add_argument("--idle-data", help="Idle data filename")
    subparser.add_argument("--machine-id", help="Machine ID")


def run():
    parser = argparse.ArgumentParser(description="S3 Pusher")
    args = util.get_command(parser, add_common_args, ibm_upload, aws_upload)
    if hasattr(args, "new_client_func") and hasattr(args, "func"):
        client = args.new_client_func(args)
        args.func(client, args.bucket_name, args.machine_id, args.mnt_path, args.query_data, args.idle_data)
    else:
        parser.print_help()


if __name__ == "__main__":
    run()
