import  argparse
import  s3.__about__ as about


def new_ibm_client(args):
    import  ibm_boto3
    from ibm_botocore.client import  Config

    cos = ibm_boto3.resource("s3", ibm_api_key_id=args.api_key, ibm_service_instance_id=args.service_instance_id, config=Config(signature_version="oauth"), endpoint_url=args.service_endpoint)
    return cos


def new_aws_client(args):
    import  boto3 as aws_boto3

    s3 = aws_boto3.client("s3", aws_access_key_id=args.aws_access_key_id, aws_secret_access_key=args.aws_secret_access_key, region_name=args.region_name)
    return s3


def get_command(parser: argparse.ArgumentParser, add_common_args, ibm_func, aws_func):
    parser.add_argument("--version", action="version", version=about.__version__)

    subparsers = parser.add_subparsers(title="S3 provider", dest="provider")
    ibm_parser = subparsers.add_parser("ibmcloud", help="IBM Cloud")
    ibm_parser.add_argument("--api-key", type=str, help="API key", required=True)
    ibm_parser.add_argument("--service-instance-id", type=str, help="Service instance ID", required=True)
    ibm_parser.add_argument("--service-endpoint", type=str, help="Service endpoint", required=True)
    add_common_args(ibm_parser)
    ibm_parser.set_defaults(new_client_func=new_ibm_client, func=ibm_func)

    aws_parser = subparsers.add_parser("aws", help="AWS")
    aws_parser.add_argument("--aws-access-key-id", type=str, help="Access key ID", required=True)
    aws_parser.add_argument("--aws-secret-access-key", type=str, help="Secret key", required=True)
    aws_parser.add_argument("--region-name", type=str, help="Region name", required=True)
    add_common_args(aws_parser)
    aws_parser.set_defaults(new_client_func=new_aws_client, func=aws_func)

    args = parser.parse_args()

    return args
