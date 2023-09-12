import json
import os
import shutil

import sys
import pandas as pd

fpath = os.path.join(os.path.dirname(__file__), 'model')
sys.path.append(fpath)

util_path = os.path.join(os.path.dirname(__file__), '..', 'util')
sys.path.append(util_path)

###############################################
# power request 

class PowerRequest():
    def __init__(self, metrics, values, output_type, source, system_features, system_values, trainer_name="", filter=""):
        self.trainer_name = trainer_name
        self.metrics = metrics
        self.filter = filter
        self.output_type = output_type
        self.energy_source = source
        self.system_features = system_features
        self.datapoint = pd.DataFrame(values, columns=metrics)
        data_point_size = len(self.datapoint)
        for i in range(len(system_features)):
            self.datapoint[system_features[i]] = [system_values[i]]*data_point_size

###############################################
# serve

import sys
import socket
import signal
from model_server_connector import ModelOutputType, make_request
from archived_model import get_achived_model
from model import load_downloaded_model
from loader import get_download_output_path
from config import set_env_from_model_config, SERVE_SOCKET, download_path
from train_types import is_support_output_type

loaded_model = dict()

def handle_request(data):
    try:
        power_request = json.loads(data, object_hook = lambda d : PowerRequest(**d))
    except Exception as e:
        msg = 'fail to handle request: {}'.format(e)
        return {"powers": dict(), "msg": msg}

    if not is_support_output_type(power_request.output_type):
        msg = "output type {} is not supported".format(power_request.output_type)
        return {"powers": dict(), "msg": msg}
    
    output_type = ModelOutputType[power_request.output_type]

    if output_type.name not in loaded_model:
        output_path = get_download_output_path(download_path, power_request.energy_source, output_type)
        if not os.path.exists(output_path):
            # try connecting to model server
            output_path = make_request(power_request)
            if output_path is None:
                # find from config
                output_path = get_achived_model(power_request)
                if output_path is None:
                    msg = "failed to get model from request {}".format(data)
                    print(msg)
                    return {"powers": dict(), "msg": msg}
                else:
                    print("load model from config: ", output_path)
            else:
                print("load model from model server: ", output_path)
        loaded_model[output_type.name] = load_downloaded_model(power_request.energy_source, output_type)
        # remove loaded model
        shutil.rmtree(output_path)

    model = loaded_model[output_type.name]
    powers, msg = model.get_power(power_request.datapoint)
    if msg != "":
        print("{} fail to predict, removed".format(model.model_name))
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
    return {"powers": powers, "msg": msg}

class EstimatorServer:
    def __init__(self, socket_path):
        self.socket_path = socket_path

    def start(self):
        s = self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(self.socket_path)
        s.listen(1)
        try:
            while True:
                connection, _ = s.accept()
                self.accepted(connection)
        finally:
            try:
                os.remove(self.socket_path)
                sys.stdout.write("close socket\n")
            except:
                pass

    def accepted(self, connection):
        data = b''
        while True:
            shunk = connection.recv(1024).strip()
            data += shunk
            if shunk is None or shunk.decode()[-1] == '}':
                break
        decoded_data = data.decode()
        y = handle_request(decoded_data)
        response = json.dumps(y)
        connection.send(response.encode())

def clean_socket():
    print("clean socket")
    if os.path.exists(SERVE_SOCKET):
        os.unlink(SERVE_SOCKET)

def sig_handler(signum, frame) -> None:
    clean_socket()
    sys.exit(1)

import argparse

if __name__ == '__main__':
    set_env_from_model_config()
    clean_socket()
    signal.signal(signal.SIGTERM, sig_handler)
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-e', '--err',
                            required=False,
                            type=str,
                            default='mae', 
                            metavar="<error metric>",
                            help="Error metric for determining the model with minimum error value" )
        args = parser.parse_args()
        DEFAULT_ERROR_KEYS = args.err.split(',')
        server = EstimatorServer(SERVE_SOCKET)
        server.start()
    finally:
        clean_socket()