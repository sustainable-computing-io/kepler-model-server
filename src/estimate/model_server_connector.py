import requests
import enum
import os
import sys
import shutil
import json
import codecs

util_path = os.path.join(os.path.dirname(__file__), '..', 'util')
sys.path.append(util_path)

from config import is_model_server_enabled, get_model_server_req_endpoint, get_model_server_list_endpoint, download_path
from loader import get_download_output_path
from train_types import ModelOutputType

def make_model_request(power_request):
    return {"metrics": power_request.metrics + power_request.system_features, "output_type": power_request.output_type, "source": power_request.energy_source, "filter": power_request.filter, "trainer_name": power_request.trainer_name}

TMP_FILE = 'tmp.zip'

def unpack(energy_source, output_type, response, replace=True):
    output_path = get_download_output_path(download_path, energy_source, output_type)
    tmp_filepath = os.path.join(download_path, TMP_FILE)
    if os.path.exists(output_path):
        if not replace:
            # delete downloaded file
            os.remove(tmp_filepath)
            return output_path
        # delete existing model
        shutil.rmtree(output_path)
    with codecs.open(tmp_filepath, 'wb') as f:
        f.write(response.content)
    shutil.unpack_archive(tmp_filepath, output_path)
    os.remove(tmp_filepath)
    return output_path

def make_request(power_request):
    if not is_model_server_enabled():
        return None
    model_request = make_model_request(power_request)
    output_type = ModelOutputType[power_request.output_type]
    try:
        response = requests.post(get_model_server_req_endpoint(), json=model_request)
    except Exception as err:
        print("cannot make request to {}: {}".format(get_model_server_req_endpoint(), err))
        return None
    if response.status_code != 200:
        return None
    return unpack(power_request.energy_source, output_type, response)

def list_all_models():
    if not is_model_server_enabled():
        return dict()
    try:
        response = requests.get(get_model_server_list_endpoint())
    except Exception as err:
        print("cannot list model: {}".format(err))
        return dict()
    if response.status_code != 200:
        return dict()
    model_names = json.loads(response.content.decode("utf-8"))
    return model_names