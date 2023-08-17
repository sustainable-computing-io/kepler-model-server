import socket
import json

import os
import sys

util_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'util')
sys.path.append(util_path)

from train_types import WORKLOAD_FEATURES, SYSTEM_FEATURES, ModelOutputType, CATEGORICAL_LABEL_TO_VOCAB
from config import SERVE_SOCKET
from extractor_test import test_energy_source

trainer_names = ['SGDRegressorTrainer']

def generate_request(train_name, n=1, metrics=WORKLOAD_FEATURES, system_features=SYSTEM_FEATURES, output_type=ModelOutputType.DynPower.name, energy_source=test_energy_source):
    request_json = dict() 
    if train_name is not None:
        request_json['trainer_name'] = train_name
    request_json['metrics'] = metrics
    request_json['system_features'] = system_features
    request_json['system_values'] = []
    for m in system_features:
        request_json['system_values'] += [CATEGORICAL_LABEL_TO_VOCAB[m][0]]
    request_json['values'] = [[1.0] *len(metrics)]*n
    request_json['output_type'] = output_type
    request_json['source'] = energy_source
    return request_json

class Client:
    def __init__(self, socket_path):
        self.socket_path = socket_path

    def make_request(self, request_json):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.socket_path)
        data = json.dumps(request_json)
        print(data)
        s.send(data.encode())
        data = b''
        while True:
            shunk = s.recv(1024).strip()
            data += shunk
            if shunk is None or len(shunk.decode()) == 0 or shunk.decode()[-1] == '}':
                break
        decoded_data = data.decode()
        s.close()
        return decoded_data

if __name__ == '__main__':
    client = Client(SERVE_SOCKET)
    request_json = generate_request(trainer_names[0], 2, output_type="AbsPower")
    res = client.make_request(request_json)
    res_json = json.loads(res)
    print(res_json)
    assert res_json["msg"]=="", "response error: {}".format(res_json["msg"])
    assert len(res_json["powers"]) > 0, "zero powers"