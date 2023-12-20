## get client
# client = new_<provider>_client(args) 
## upload all files in mnt path 
# <provider>_upload(client, mnt_path) 

def new_ibm_client(args):
    import ibm_boto3
    from ibm_botocore.client import Config
    cos = ibm_boto3.resource('s3',
        ibm_api_key_id=args.api_key,
        ibm_service_instance_id=args.service_instance_id,
        config=Config(signature_version='oauth'),
        endpoint_url=args.service_endpoint
    )
    return cos
    
def new_aws_client(args):
    import boto3 as aws_boto3
    s3 = aws_boto3.client('s3', aws_access_key_id=args.aws_access_key_id, aws_secret_access_key=args.aws_secret_access_key, region_name=args.region_name)
    return s3

model_dir="models"
data_dir="data"
import os
def get_bucket_file_map(machine_id, mnt_path, query_data, idle_data):
    model_path = os.path.join(mnt_path, model_dir)
    bucket_file_map = dict()
    top_key_path = ""
    if machine_id is not None and machine_id != "":
        top_key_path = "/" + machine_id
    for root, _, files in os.walk(model_path):
        for file in files:
            filepath = os.path.join(root, file)
            key = filepath.replace(model_path, top_key_path + "/models")
            bucket_file_map[key] = filepath
    data_path = os.path.join(mnt_path, data_dir)
    for data_filename in [query_data, idle_data]:
        if data_filename is not None:
            filepath = os.path.join(data_path, data_filename + ".json")
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

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S3 Pusher")
    subparsers = parser.add_subparsers(title="S3 provider", dest="provider")

    ibm_parser = subparsers.add_parser("ibmcloud", help="IBM Cloud")
    ibm_parser.add_argument("--api-key", type=str, help="API key", required=True)
    ibm_parser.add_argument("--service-instance-id", type=str, help="Service instance ID", required=True)
    ibm_parser.add_argument("--service-endpoint", type=str, help="Service endpoint", required=True)
    add_common_args(ibm_parser)
    ibm_parser.set_defaults(new_client_func=new_ibm_client, upload_func=ibm_upload)

    aws_parser = subparsers.add_parser("aws", help="AWS")
    aws_parser.add_argument("--aws-access-key-id", type=str, help="Access key ID", required=True)
    aws_parser.add_argument("--aws-secret-access-key", type=str, help="Secret key", required=True)
    aws_parser.add_argument("--region-name", type=str, help="Region name", required=True)
    add_common_args(aws_parser)
    aws_parser.set_defaults(new_client_func=new_aws_client, upload_func=aws_upload)

    args = parser.parse_args()

    if hasattr(args, "new_client_func") and hasattr(args, "upload_func"):
        client = args.new_client_func(args)
        args.upload_func(client, args.bucket_name, args.machine_id, args.mnt_path, args.query_data, args.idle_data)
    else:
        parser.print_help()