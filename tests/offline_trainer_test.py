# offline_trainer_test.py
#
# client (test):
#   python tests/offline_trainer_test.py dataset_name <src_json_file> <idle_filepath> <save_path>
# output will be saved at
# save_path |- dataset_name |- AbsPower |- power_source |- feature_group |- metadata.json, ...
#                           |- DynPower |- power_source |- feature_group |- metadata.json, ...
# test offline trainer
#

import requests

import os
import sys
import shutil
import json
import codecs

src_path = os.path.join(os.path.dirname(__file__), '../src')
train_path = os.path.join(os.path.dirname(__file__), '../src/train')
profile_path = os.path.join(os.path.dirname(__file__), '../src/profile')
util_path = os.path.join(os.path.dirname(__file__), 'util')

sys.path.append(src_path)
sys.path.append(train_path)
sys.path.append(profile_path)
sys.path.append(util_path)

from train.offline_trainer import TrainAttribute, TrainRequest, serve_port


from model_server_test import TMP_FILE
from profile import profile_process
from extractor_test import prom_response_file, prom_response_idle_file

from train.prom.prom_query import prom_responses_to_results

offline_trainer_output_path = os.path.join(os.path.dirname(__file__), 'data', 'offline_trainer_output')

if not os.path.exists(offline_trainer_output_path):
    os.mkdir(offline_trainer_output_path)

target_suffix = "_True.csv"

abs_trainer_names = ['GradientBoostingRegressorTrainer', 'SGDRegressorTrainer', 'KNeighborsRegressorTrainer', 'LinearRegressionTrainer', 'PolynomialRegressionTrainer', 'SVRRegressorTrainer']
dyn_trainer_names = ['GradientBoostingRegressorTrainer', 'SGDRegressorTrainer', 'KNeighborsRegressorTrainer', 'LinearRegressionTrainer', 'PolynomialRegressionTrainer', 'SVRRegressorTrainer']

energy_sources = ['rapl']
feature_groups = ['CounterOnly', 'CgroupOnly', 'BPFOnly', 'KubeletOnly']

profiles = dict()

isolators = {
    'MinIdleIsolator': [],
    'ProfileBackgroundIsolator': [profiles],
    'TrainIsolator': [abs_trainer_names] # apply all and select the best
}


def get_target_path(save_path, energy_source, feature_group):
    power_path = os.path.join(save_path, energy_source)
    if not os.path.exists(power_path):
        os.mkdir(power_path)
    feature_path = os.path.join(save_path, feature_group)
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
    return feature_path

import json

def make_request(dataset_name, profiles, isolator, isolator_args, data, save_path):
    trainer = TrainAttribute(abs_trainer_names, dyn_trainer_names, profiles, isolator, isolator_args)
    request = TrainRequest(dataset_name, trainer=trainer, data=data)

    # send request
    response = requests.post('http://localhost:{}/train'.format(serve_port), json=request)
    assert response.status_code == 200, response.text
    with codecs.open(TMP_FILE, 'wb') as f:
        f.write(response.content)
    # unpack response
    shutil.unpack_archive(TMP_FILE, save_path)
    os.remove(TMP_FILE)

def get_pipeline_name(dataset_name, isolator):
    return "{}_{}".format(dataset_name, isolator)  

def process(dataset_name, json_file_path, idle_file_path, target_path):
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    with open(idle_file_path) as f:
        idle_response = json.load(f)
        idle_data = prom_responses_to_results(idle_response)
    with open(json_file_path) as f:
        train_response = json.load(f)
        train_data = prom_responses_to_results(train_response)
    profiles = profile_process(idle_data)
    for isolator, isolator_args in isolators.items():
        print("Isolator: ", isolator)
        pipeline_name = get_pipeline_name(dataset_name, isolator)
        save_path = os.path.join(target_path, pipeline_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        make_request(pipeline_name, profiles, isolator, isolator_args, train_data, save_path)
    
                
if __name__ == '__main__':
    dataset_name = "sample_data"
    json_file_path = prom_response_file
    idle_file_path = prom_response_idle_file
    target_path = offline_trainer_output_path

    process(dataset_name, json_file_path, idle_file_path, target_path)
    