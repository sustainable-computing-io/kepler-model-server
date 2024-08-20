import requests
import os
import shutil
import json
import codecs


from kepler_model.util.config import is_model_server_enabled, get_model_server_req_endpoint, get_model_server_list_endpoint, download_path
from kepler_model.util.loader import get_download_output_path
from kepler_model.util.train_types import ModelOutputType
from kepler_model.train.profiler.node_type_index import discover_spec_values

machine_spec_mount_path = "/etc/kepler/models/machine_spec"
machine_id = os.getenv('MACHINE_ID', None)

# get_spec_values: determine node spec in json format (refer to NodeTypeSpec)
def get_spec_values(machine_id : str|None):
    if machine_id is not None:
        spec_file = os.path.join(machine_spec_mount_path, machine_id)
        try:
            with open(spec_file) as f:
                res = json.load(f)
            return res
        except:
            pass
    return discover_spec_values()

node_spec = None


def make_model_request(power_request):
    global node_spec
    if node_spec is None:
         node_spec = get_spec_values(machine_id)
         print(f"Node spec: {node_spec}")
    return {"metrics": power_request.metrics + power_request.system_features, "output_type": power_request.output_type, "source": power_request.energy_source, "filter": power_request.filter, "trainer_name": power_request.trainer_name, "spec": node_spec}


TMP_FILE = "tmp.zip"


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
        print("cannot make request to {}: {}".format(get_model_server_req_endpoint(), err))
        return None
    if response.status_code != 200:
        return None
    return unpack(power_request.energy_source, output_type, response)


def list_all_models(energy_source=None, node_type=None):
    if not is_model_server_enabled():
        return dict()
    try:
        endpoint = get_model_server_list_endpoint()
        params= {}
        if energy_source:
            params["source"] = energy_source
        if node_type:
           params["type"] = node_type
        response = requests.get(endpoint, params=params)
    except Exception as err:
        print("cannot list model: {}".format(err))
        return dict()
    if response.status_code != 200:
        return dict()
    model_names = json.loads(response.content.decode("utf-8"))
    return model_names
