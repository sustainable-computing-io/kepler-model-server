"""
python src/train/ec2_pipeline.py

This program trains a pipeline called `ec2` from the collected data on AWS COS which is collected by kepler-model-training-playbook or by collect-data job on github workflow.

required step:
- set AWS secret environments (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
"""

import os
import sys
import json

from profiler.node_type_index import NodeTypeSpec, NodeAttribute

cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)
util_path = os.path.join(os.path.dirname(__file__), '..', 'util')
sys.path.append(util_path)
extractor_path = os.path.join(os.path.dirname(__file__), 'extractor')
sys.path.append(extractor_path)
isolator_path = os.path.join(os.path.dirname(__file__), 'isolator')
sys.path.append(isolator_path)
profiler_path = os.path.join(os.path.dirname(__file__), 'profiler')
sys.path.append(profiler_path)

from pipeline import NewPipeline
from extractor import DefaultExtractor
from isolator import MinIdleIsolator
from prom_types import node_info_column, prom_responses_to_results, get_valid_feature_group_from_queries
from train_types import default_trainer_names, PowerSourceMap

node_profiles = ["m5zn.metal", "c5d.metal", "i3en.metal", "m7i.metal-24xl", "i3.metal"]
node_image = "ami-0e4d0bb9670ea8db0"

import boto3

aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
region_name = os.environ["AWS_REGION"]

# Initialize the S3 client
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)
bucket_name = 'kepler-power-model'
unknown = -1
def read_response_in_json(key):
    print(key)
    response = s3.get_object(Bucket=bucket_name, Key=key)
    return json.loads(response['Body'].read().decode('utf-8'))

class Ec2PipelineRun():
    def __init__(self, name, abs_trainer_names=default_trainer_names, dyn_trainer_names=default_trainer_names, isolator=MinIdleIsolator()):
        self.energy_source = "intel_rapl"
        self.energy_components = PowerSourceMap[self.energy_source]
        self.pipeline = NewPipeline(name, abs_trainer_names=abs_trainer_names, dyn_trainer_names=dyn_trainer_names, extractor=DefaultExtractor(), isolator=isolator, target_energy_sources=[self.energy_source])

    def process(self):
        for profile in node_profiles:
            machine_id = "-".join([profile, node_image])
            kepler_query_key = os.path.join("/", machine_id, "data", "kepler_query.json")
            spec_key = os.path.join("/", machine_id, "data", "machine_spec", machine_id + ".json")
            query_response = read_response_in_json(kepler_query_key)
            spec_json = read_response_in_json(spec_key)
            spec = NodeTypeSpec(**spec_json['attrs'])
            spec.attrs[NodeAttribute.PROCESSOR] = profile
            spec.attrs[NodeAttribute.MEMORY] = unknown
            spec.attrs[NodeAttribute.FREQ] = unknown
            node_type = self.pipeline.node_collection.index_train_machine(machine_id, spec)
            query_results = prom_responses_to_results(query_response)
            query_results[node_info_column] = node_type
            valid_fg = get_valid_feature_group_from_queries([query for query in query_response.keys() if len(query_response[query]) > 1 ])
            for feature_group in valid_fg:
                self.pipeline.process(query_results, self.energy_components, self.energy_source, feature_group=feature_group.name, replace_node_type=node_type)
        self.pipeline.node_collection.save()

    def save_metadata(self):
        self.pipeline.save_metadata()

    def archive_pipeline(self):
        self.pipeline.archive_pipeline()

if __name__ == "__main__":
    pipeline_name = "ec2"
    pipelinerun = Ec2PipelineRun(name=pipeline_name)
    pipelinerun.process()
    pipelinerun.save_metadata()
    pipelinerun.archive_pipeline()