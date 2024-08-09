"""
python src/train/ec2_pipeline.py

This program trains a pipeline called `ec2` from the collected data on AWS COS which is collected by kepler-model-training-playbook or by collect-data job on github workflow.

before run:
- set AWS secret environments (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)

after run:
example of plot command:
DATAPATH=/path/to/models python cmd/main.py plot --target-data estimate --input /path/to/data/i3.metal/kepler_query.json --pipeline-name ec2 --energy-source rapl-sysfs --model-name LinearRegressionTrainer_4 --output-type AbsPower --output i3metal-ec2 --feature-group BPFOnly
DATAPATH=/path/to/models python cmd/main.py plot --target-data estimate --input /path/to/data/i3.metal/kepler_query.json --pipeline-name ec2 --energy-source rapl-sysfs --model-name LogarithmicRegressionTrainer_4 --output-type AbsPower --output i3metal-ec2 --feature-group BPFOnly

example of export command:
DATAPATH=/path/to/models python cmd/main.py export --pipeline-name ec2-0.7.11 -o /path/to/kepler-model-db/models --publisher sunya-ch --zip=true --collect-date "July 2024"
"""

import os
import json
import boto3

from kepler_model.train.profiler.node_type_index import NodeTypeSpec, NodeAttribute
from kepler_model.train.pipeline import NewPipeline
from kepler_model.train.extractor import DefaultExtractor
from kepler_model.train.isolator.isolator import MinIdleIsolator
from kepler_model.util.prom_types import node_info_column, prom_responses_to_results, get_valid_feature_group_from_queries

from kepler_model.util.train_types import default_trainer_names, PowerSourceMap
from kepler_model.util.saver import save_json
from kepler_model.util.config import model_toppath

data_path = os.path.join(model_toppath, "..", "data")

node_profiles = ["m5.metal", "i3.metal", "c5.metal", "r5.metal", "m5zn.metal", "m7i.metal-24xl"]
node_image = "ami-0e4d0bb9670ea8db0"


last_modified = None

aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
region_name = os.environ["AWS_REGION"]

# Initialize the S3 client
s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)
bucket_name = "kepler-power-model"
unknown = -1


def read_response_in_json(key):
    print(key)
    response = s3.get_object(Bucket=bucket_name, Key=key)
    global last_modified
    last_modified = response["LastModified"]
    print("{} last modified time: {}".format(key, last_modified))
    return json.loads(response["Body"].read().decode("utf-8"))


class Ec2PipelineRun:
    def __init__(self, name, abs_trainer_names=default_trainer_names, dyn_trainer_names=default_trainer_names, isolator=MinIdleIsolator()):
        self.energy_source = "rapl-sysfs"
        self.energy_components = PowerSourceMap[self.energy_source]
        self.pipeline = NewPipeline(name, abs_trainer_names=abs_trainer_names, dyn_trainer_names=dyn_trainer_names, extractor=DefaultExtractor(), isolator=isolator, target_energy_sources=[self.energy_source])

    def process(self):
        for profile in node_profiles:
            # load data from COS
            machine_id = "-".join([profile, node_image])
            kepler_query_key = os.path.join("/", machine_id, "data", "kepler_query.json")
            machine_spec_filename = machine_id + ".json"
            spec_key = os.path.join("/", machine_id, "data", "machine_spec", machine_spec_filename)
            query_response = read_response_in_json(kepler_query_key)
            spec_json = read_response_in_json(spec_key)

            # save raw data from response in local path models/../data/<instance profile>
            profile_datapath = os.path.join(data_path, profile)
            os.makedirs(profile_datapath, exist_ok=True)
            save_json(path=profile_datapath, name="kepler_query.json", data=query_response)
            machine_spec_path = os.path.join(profile_datapath, "machine_spec")
            save_json(path=machine_spec_path, name=machine_spec_filename, data=spec_json)

            # process pipeline
            spec = NodeTypeSpec()
            spec.load(spec_json)
            node_type = self.pipeline.node_collection.index_train_machine(machine_id, spec)
            query_results = prom_responses_to_results(query_response)
            query_results[node_info_column] = node_type
            valid_fg = get_valid_feature_group_from_queries([query for query in query_response.keys() if len(query_response[query]) > 1])
            for feature_group in valid_fg:
                self.pipeline.process(query_results, self.energy_components, self.energy_source, feature_group=feature_group.name, replace_node_type=node_type)
        self.pipeline.node_collection.save()

    def save_metadata(self):
        self.pipeline.save_metadata()

    def archive_pipeline(self):
        self.pipeline.archive_pipeline()


if __name__ == "__main__":
    pipeline_name = "ec2-0.7.11"
    pipelinerun = Ec2PipelineRun(name=pipeline_name)
    pipelinerun.process()
    pipelinerun.save_metadata()
    pipelinerun.archive_pipeline()
    print("Collection time:", last_modified)
    item = dict()
    item["startTimeUTC"] = last_modified.strftime("%Y-%m-%dT%H:%M:%SZ")
    item["endTimeUTC"] = last_modified.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(item)
    save_json(path=data_path, name=pipeline_name, data=item)

