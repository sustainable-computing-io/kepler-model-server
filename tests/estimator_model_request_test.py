#########################
# estimator_test.py
# 
# This file covers the following cases.
# - kepler-model-server is connected
#   - list all available models with its corresponding available feature groups and make a dummy PowerRequest
# - kepler-model-server is not connected, but some achived models can be download via URL.
#   - set sample model and make a dummy valid PowerRequest and another invalid PowerRequest
#
#########################
# import external modules
import shutil

# import from src
import os
import sys

server_path = os.path.join(os.path.dirname(__file__), '../src')
util_path = os.path.join(os.path.dirname(__file__), '../src/util')
train_path = os.path.join(os.path.dirname(__file__), '../src/train')
estimate_path = os.path.join(os.path.dirname(__file__), '../src/estimate')
prom_path = os.path.join(os.path.dirname(__file__), '../src/train/prom')

sys.path.append(server_path)
sys.path.append(util_path)
sys.path.append(train_path)
sys.path.append(prom_path)
sys.path.append(estimate_path)

from train_types import FeatureGroups, FeatureGroup, ModelOutputType
from loader import get_download_output_path
from estimate.estimator import handle_request, loaded_model, PowerRequest
from estimate.model_server_connector import list_all_models
from estimate.archived_model import get_achived_model, reset_failed_list
from config import getConfig, estimatorKeyMap, initUrlKeyMap, set_env_from_model_config, get_url

from estimator_power_request_test import generate_request

os.environ['MODEL_SERVER_URL'] = 'http://localhost:8100'

import json

if __name__ == '__main__':
    # test getting model from server
    os.environ['MODEL_SERVER_ENABLE'] = "true"
    available_models = list_all_models()
    print(available_models)
    for output_type_name, valid_fgs in available_models.items():
        if 'Weight' in output_type_name:
            continue
        output_type = ModelOutputType[output_type_name]
        output_path = get_download_output_path(output_type)
        for fg_name, best_model in valid_fgs.items():
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            if output_type.name in loaded_model:
                del loaded_model[output_type.name]
            metrics = FeatureGroups[FeatureGroup[fg_name]]
            request_json = generate_request(None, n=10, metrics=metrics, output_type=output_type_name)
            data = json.dumps(request_json)
            output = handle_request(data)
            assert len(output['powers']) > 0, "cannot get power {}\n {}".format(output['msg'], request_json)
            print("result {}/{} from model server: {}".format(output_type_name, fg_name, output))

    # test with initial models
    os.environ['MODEL_SERVER_ENABLE'] = "false"
    for output_type in ModelOutputType:
        output_type_name = output_type.name
        if output_type_name not in initUrlKeyMap:
            continue
        url = getConfig(initUrlKeyMap[output_type_name], None)
        if url is not None:
            output_path = get_download_output_path(output_type)
            if output_type_name in loaded_model:
                del loaded_model[output_type_name]
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            request_json = generate_request(None, n=10, metrics=FeatureGroups[FeatureGroup.Full], output_type=output_type_name)
            data = json.dumps(request_json)
            output = handle_request(data)
            assert len(output['powers']) > 0, "cannot get power {}\n {}".format(output['msg'], request_json)
            print("result from {}: {}".format(url, output))

    # test getting model from archived
    os.environ['MODEL_SERVER_ENABLE'] = "false"
    if len(available_models) == 0:
        output_type_name = 'DynPower'
        # enable model to use
        os.environ[estimatorKeyMap[output_type_name]] = "true"
        output_type = ModelOutputType[output_type_name]
        output_path = get_download_output_path(output_type)
        if output_type_name in loaded_model:
            del loaded_model[output_type_name]
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        # valid model
        os.environ[initUrlKeyMap[output_type_name]] = get_url(output_type=output_type, feature_group=FeatureGroup.CgroupOnly)
        request_json = generate_request(None, n=10, metrics=FeatureGroups[FeatureGroup.CgroupOnly], output_type=output_type_name)
        data = json.dumps(request_json)
        output = handle_request(data)
        assert len(output['powers']) > 0, "cannot get power {}\n {}".format(output['msg'], request_json)
        print("result {}/{} from static set: {}".format(output_type_name, FeatureGroup.CgroupOnly.name, output))
        del loaded_model[output_type_name]
        # invalid model
        os.environ[initUrlKeyMap[output_type_name]] = get_url(output_type=output_type, feature_group=FeatureGroup.BPFOnly)
        request_json = generate_request(None, n=10, metrics=FeatureGroups[FeatureGroup.CgroupOnly], output_type=output_type_name)
        data = json.dumps(request_json)
        power_request = json.loads(data, object_hook = lambda d : PowerRequest(**d))
        output_path = get_achived_model(power_request)
        assert output_path is None, "model should be invalid\n {}".format(output_path)
        os.environ['MODEL_CONFIG'] = "CONTAINER_COMPONENTS_ESTIMATOR=true\nCONTAINER_COMPONENTS_INIT_URL={}\n".format(get_url(output_type=output_type, feature_group=FeatureGroup.CgroupOnly))
        set_env_from_model_config()
        reset_failed_list()
        if output_type_name in loaded_model:
            del loaded_model[output_type_name]
        output_path = get_download_output_path(output_type)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        request_json = generate_request(None, n=10, metrics=FeatureGroups[FeatureGroup.CgroupOnly], output_type=output_type_name)
        data = json.dumps(request_json)
        output = handle_request(data)
        assert len(output['powers']) > 0, "cannot get power {}\n {}".format(output['msg'], request_json)