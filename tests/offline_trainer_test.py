# offline_trainer_test.py
#
# client (test):
#   python tests/offline_trainer_test.py dataset_name <src_json_file> <idle_filepath> <save_path>
# output will be saved at
# save_path |- dataset_name |- AbsPower |- energy_source |- feature_group |- metadata.json, ...
#                           |- DynPower |- energy_source |- feature_group |- metadata.json, ...
# test offline trainer
#

import codecs
import os
import shutil

import requests

from kepler_model.train.offline_trainer import TrainAttribute, TrainRequest, serve_port
from kepler_model.util.loader import class_to_json, list_all_abs_models, list_all_dyn_models
from kepler_model.util.prom_types import get_valid_feature_group_from_queries, prom_responses_to_results
from tests.extractor_test import test_energy_source
from tests.model_server_test import TMP_FILE
from tests.prom_test import get_prom_response

offline_trainer_output_path = os.path.join(os.path.dirname(__file__), "data", "offline_trainer_output")

if not os.path.exists(offline_trainer_output_path):
    os.mkdir(offline_trainer_output_path)

# example trainers
abs_trainer_names = ["SGDRegressorTrainer", "PolynomialRegressionTrainer"]
dyn_trainer_names = ["SGDRegressorTrainer", "PolynomialRegressionTrainer"]

# requested isolators
isolators = {
    "MinIdleIsolator": {},
    "NoneIsolator": {},
    "ProfileBackgroundIsolator": {},
    # "TrainIsolator": {"abs_pipeline_name": default_train_output_pipeline} # TODO: too heavy to test on CI
}


def get_target_path(save_path, energy_source, feature_group):
    power_path = os.path.join(save_path, energy_source)
    if not os.path.exists(power_path):
        os.mkdir(power_path)
    feature_path = os.path.join(save_path, feature_group)
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
    return feature_path


def make_request(pipeline_name, idle_prom_response, isolator, isolator_args, prom_response, energy_source, save_path):
    trainer = TrainAttribute(abs_trainer_names, dyn_trainer_names, idle_prom_response, isolator, isolator_args)
    # set trainer to None to not initialize TrainAttribute object
    train_request = TrainRequest(pipeline_name, trainer=None, prom_response=prom_response, energy_source=energy_source)
    train_request.trainer = class_to_json(trainer)
    request = class_to_json(train_request)
    # send request
    response = requests.post(f"http://localhost:{serve_port}/train", json=request)
    assert response.status_code == 200, response.text
    with codecs.open(TMP_FILE, "wb") as f:
        f.write(response.content)
    # unpack response
    shutil.unpack_archive(TMP_FILE, save_path)
    os.remove(TMP_FILE)


def get_pipeline_name(dataset_name, isolator):
    return f"{dataset_name}_{isolator}"


def _assert_offline_trainer(model_list_map):
    for model_path, models in model_list_map.items():
        assert len(models) > 0, f"No trained model in {model_path}"
        print(f"Trained model in {model_path}: {models}")


def assert_offline_trainer_output(target_path, energy_source, valid_fgs, pipeline_name):
    abs_models = list_all_abs_models(target_path, energy_source, valid_fgs, pipeline_name=pipeline_name)
    dyn_models = list_all_dyn_models(target_path, energy_source, valid_fgs, pipeline_name=pipeline_name)
    _assert_offline_trainer(abs_models)
    _assert_offline_trainer(dyn_models)


def process(dataset_name, train_prom_response, idle_prom_response, energy_source=test_energy_source, isolators=isolators, target_path=offline_trainer_output_path):
    idle_data = prom_responses_to_results(idle_prom_response)
    valid_fgs = get_valid_feature_group_from_queries(idle_data)

    for isolator, isolator_args in isolators.items():
        print("Isolator: ", isolator)
        pipeline_name = get_pipeline_name(dataset_name, isolator)
        save_path = os.path.join(target_path, pipeline_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        make_request(pipeline_name, idle_prom_response, isolator, isolator_args, train_prom_response, energy_source, save_path)
        assert_offline_trainer_output(target_path=target_path, energy_source=energy_source, valid_fgs=valid_fgs, pipeline_name=pipeline_name)


def test_offline_trainer():
    dataset_name = "sample_data"
    idle_prom_response = get_prom_response(save_name="idle")
    train_prom_response = get_prom_response()
    process(dataset_name, train_prom_response, idle_prom_response)
