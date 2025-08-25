import json
import socket

from kepler_model.util.config import SERVE_SOCKET
from kepler_model.util.train_types import (
    CATEGORICAL_LABEL_TO_VOCAB,
    SYSTEM_FEATURES,
    BPF_FEATURES,
    ModelOutputType,
)
from tests.extractor_test import test_energy_source

trainer_names = ["SGDRegressorTrainer_0", "LogarithmicRegressionTrainer_0.json"]
test_energy_sources = ["acpi", "rapl-sysfs"]


def generate_request(
    train_name, n=1, metrics=BPF_FEATURES, system_features=SYSTEM_FEATURES, output_type=ModelOutputType.AbsPower.name, energy_source=test_energy_source
):
    request_json = dict()
    if train_name is not None:
        request_json["trainer_name"] = train_name
    request_json["metrics"] = metrics
    request_json["system_features"] = system_features
    request_json["system_values"] = []
    for m in system_features:
        request_json["system_values"] += [CATEGORICAL_LABEL_TO_VOCAB[m][0]]
    request_json["values"] = [[1000.0] * len(metrics)] * n
    request_json["output_type"] = output_type
    request_json["source"] = energy_source
    return request_json


def process(client, energy_source):
    for trainer_name in trainer_names:
        request_json = generate_request(trainer_name, 2, output_type="AbsPower", energy_source=energy_source)
        res = client.make_request(request_json)
        res_json = json.loads(res)
        print(res_json)
        assert res_json["msg"] == "", "response error: {}".format(res_json["msg"])
        assert len(res_json["powers"]) > 0, "zero powers"


class Client:
    def __init__(self, socket_path):
        self.socket_path = socket_path

    def make_request(self, request_json):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.socket_path)
        data = json.dumps(request_json)
        print(data)
        s.send(data.encode())
        data = b""
        while True:
            shunk = s.recv(1024).strip()
            data += shunk
            if shunk is None or len(shunk.decode()) == 0 or shunk.decode()[-1] == "}":
                break
        decoded_data = data.decode()
        s.close()
        return decoded_data


def test_estimator_power_request():
    client = Client(SERVE_SOCKET)
    for energy_source in test_energy_sources:
        process(client, energy_source)


if __name__ == "__main__":
    test_estimator_power_request()
