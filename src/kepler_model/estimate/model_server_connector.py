import codecs
import json
import os
import shutil

import requests

from kepler_model.server.model_server import ModelListParam
from kepler_model.util.config import (
    download_path,
    get_model_server_list_endpoint,
    get_model_server_req_endpoint,
    is_model_server_enabled,
)
from kepler_model.util.loader import get_download_output_path
from kepler_model.util.train_types import ModelOutputType


# discover_spec: determine node spec in json format (refer to NodeTypeSpec)
def discover_spec():
    import psutil

    # TODO: reuse node_type_index/generate_spec with loosen selection
    cores = psutil.cpu_count(logical=True)
    spec = {"cores": cores}
    return spec


node_spec = discover_spec()


def make_model_request(power_request):
    return {"metrics": power_request.metrics + power_request.system_features, "output_type": power_request.output_type, "source": power_request.energy_source, "filter": power_request.filter, "trainer_name": power_request.trainer_name, "spec": node_spec}


TMP_FILE = "tmp.zip"


def unpack(energy_source, output_type, response, replace=True):
    output_path = get_download_output_path(download_path, energy_source, output_type)
    tmp_filepath = os.path.join(download_path, TMP_FILE)
    if os.path.exists(output_path):
        if not replace:
            if os.path.exists(tmp_filepath):
                # delete downloaded file
                os.remove(tmp_filepath)
            return output_path
        # delete existing model
        shutil.rmtree(output_path)
    with codecs.open(tmp_filepath, "wb") as f:
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
        print(f"cannot make request to {get_model_server_req_endpoint()}: {err}")
        return None
    if response.status_code != 200:
        return None
    return unpack(power_request.energy_source, output_type, response)


def list_all_models(energy_source=None, output_type=None, feature_group=None, node_type=None, filter=None):
    if not is_model_server_enabled():
        return dict()
    try:
        endpoint = get_model_server_list_endpoint()
        params= {}
        if energy_source:
            params[ModelListParam.EnergySource.value] = energy_source
        if output_type:
            params[ModelListParam.OutputType.value] = output_type
        if feature_group:
            params[ModelListParam.FeatureGroup.value] = feature_group
        if node_type:
           params[ModelListParam.NodeType.value] = node_type
        if filter:
            params[ModelListParam.Filter.value] = filter

        response = requests.get(endpoint, params=params)
    except Exception as err:
        print(f"cannot list model: {err}")
        return dict()
    if response.status_code != 200:
        return dict()
    model_names = json.loads(response.content.decode("utf-8"))
    return model_names
