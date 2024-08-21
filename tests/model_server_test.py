import codecs
import json
import os
import shutil

import requests

from kepler_model.server.model_server import MODEL_SERVER_PORT
from kepler_model.util.config import download_path
from kepler_model.util.train_types import FeatureGroup, FeatureGroups, ModelOutputType

TMP_FILE = "tmp.zip"


def get_model_request_json(metrics, output_type, node_type, weight, trainer_name, energy_source):
    return {"metrics": metrics, "output_type": output_type.name, "node_type": node_type, "weight": weight, "trainer_name": trainer_name, "source": energy_source}


def make_request(metrics, output_type, node_type=-1, weight=False, trainer_name="", energy_source="rapl-sysfs"):
    model_request = get_model_request_json(metrics, output_type, node_type, weight, trainer_name, energy_source)
    response = requests.post(f"http://localhost:{MODEL_SERVER_PORT}/model", json=model_request)
    assert response.status_code == 200, response.text
    if weight:
        weight_dict = json.loads(response.text)
        assert len(weight_dict) > 0, "weight dict must contain one or more than one component"
        for weight_values in weight_dict.values():
            weight_length = len(weight_values["All_Weights"]["Numerical_Variables"])
            expected_length = len(metrics)
            assert weight_length <= expected_length, f"weight metrics should covered by the requested {weight_length} > {expected_length}"
    else:
        output_path = os.path.join(download_path, output_type.name)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        with codecs.open(TMP_FILE, "wb") as f:
            f.write(response.content)
        shutil.unpack_archive(TMP_FILE, output_path)
        os.remove(TMP_FILE)


def get_models():
    response = requests.get(f"http://localhost:{MODEL_SERVER_PORT}/best-models")
    assert response.status_code == 200, response.text
    response = json.loads(response.text)
    return response


if __name__ == "__main__":
    models = get_models()
    assert len(models) > 0, "more than one type of output"
    for output_models in models.values():
        assert len(output_models) > 0, "more than one best model for each output"

    test_feature_groups = [FeatureGroup.BPFOnly, FeatureGroup.CounterOnly]

    # for each features
    for fg in test_feature_groups:
        metrics = FeatureGroups[fg]
        # abs power
        output_type = ModelOutputType.AbsPower
        make_request(metrics, output_type)
        make_request(metrics, output_type, weight=True)
        # dyn power
        output_type = ModelOutputType.DynPower
        make_request(metrics, output_type)
        make_request(metrics, output_type, weight=True)

    metrics = FeatureGroups[FeatureGroup.BPFOnly]
    # with node_type
    make_request(metrics, output_type, node_type=1)
    make_request(metrics, output_type, node_type=1, weight=True)
    # with trainer name
    trainer_name = "SGDRegressorTrainer"
    make_request(metrics, output_type, trainer_name=trainer_name)
    make_request(metrics, output_type, trainer_name=trainer_name, node_type=1)
    make_request(metrics, output_type, trainer_name=trainer_name, node_type=1, weight=True)
    # with acpi source
    make_request(metrics, output_type, energy_source="acpi", trainer_name=trainer_name, node_type=1, weight=True)
