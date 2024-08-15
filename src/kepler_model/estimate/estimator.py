import json
import os
import shutil
import sys
import argparse
import logging

import pandas as pd

import socket
import signal
from kepler_model.estimate.model_server_connector import make_request
from kepler_model.estimate.archived_model import get_achived_model
from kepler_model.estimate.model.model import load_downloaded_model
from kepler_model.util.loader import get_download_output_path
from kepler_model.util.config import set_env_from_model_config, SERVE_SOCKET, download_path
from kepler_model.util.train_types import is_support_output_type, ModelOutputType

###############################################
# power request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PowerRequest:
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
            self.datapoint[system_features[i]] = [system_values[i]] * data_point_size


###############################################
# serve


loaded_model = dict()


def handle_request(data):
    try:
        power_request = json.loads(data, object_hook=lambda d: PowerRequest(**d))
    except Exception as e:
        logger.error("fail to handle request: {}".format(e))
        msg = "fail to handle request: {}".format(e)
        return {"powers": dict(), "msg": msg}

    if not is_support_output_type(power_request.output_type):
        msg = "output type {} is not supported".format(power_request.output_type)
        logger.error(msg)
        return {"powers": dict(), "msg": msg}

    output_type = ModelOutputType[power_request.output_type]
    # TODO: need revisit if get more than one rapl energy source
    if power_request.energy_source is None or "rapl" in power_request.energy_source:
        power_request.energy_source = "rapl-sysfs"

    if output_type.name not in loaded_model:
        loaded_model[output_type.name] = dict()
    output_path = ""
    request_trainer = False
    if power_request.trainer_name is not None:
        if output_type.name in loaded_model and power_request.energy_source in loaded_model[output_type.name]:
            current_trainer = loaded_model[output_type.name][power_request.energy_source].trainer_name
            request_trainer = current_trainer != power_request.trainer_name
            if request_trainer:
                logger.info("try obtaining the requesting trainer {} (current: {})".format(power_request.trainer_name, current_trainer))
    if power_request.energy_source not in loaded_model[output_type.name] or request_trainer:
        output_path = get_download_output_path(download_path, power_request.energy_source, output_type)
        if not os.path.exists(output_path):
            # try connecting to model server
            output_path = make_request(power_request)
            if output_path is None:
                # find from config
                output_path = get_achived_model(power_request)
                if output_path is None:
                    msg = "failed to get model from request {}".format(data)
                    logger.error(msg)
                    return {"powers": dict(), "msg": msg}
                else:
                    logger.info("load model from config: ", output_path)
            else:
                logger.info("load model from model server: %s", output_path)
        loaded_item = load_downloaded_model(power_request.energy_source, output_type)
        if loaded_item is not None and loaded_item.estimator is not None:
            loaded_model[output_type.name][power_request.energy_source] = loaded_item
            logger.info("set model {0} for {2} ({1})".format(loaded_item.model_name, output_type.name, power_request.energy_source))
        # remove loaded model
        shutil.rmtree(output_path)

    model = loaded_model[output_type.name][power_request.energy_source]
    powers, msg = model.get_power(power_request.datapoint)
    if msg != "":
        logger.info("{} fail to predict, removed: {}".format(model.model_name, msg))
        if output_path != "" and os.path.exists(output_path):
            shutil.rmtree(output_path)
    return {"powers": powers, "msg": msg}


class EstimatorServer:
    def __init__(self, socket_path):
        self.socket_path = socket_path

    def start(self):
        s = self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(self.socket_path)
        s.listen(1)
        logger.info("started serving on {}".format(self.socket_path))
        try:
            while True:
                connection, _ = s.accept()
                self.accepted(connection)
        finally:
            try:
                os.remove(self.socket_path)
                sys.stdout.write("close socket\n")
            except Exception as e:
                logger.error("fail to close socket: ", e)
                pass

    def accepted(self, connection):
        data = b""
        while True:
            shunk = connection.recv(1024).strip()
            data += shunk
            if shunk is None or shunk.decode()[-1] == "}":
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


def run():
    set_env_from_model_config()
    clean_socket()
    signal.signal(signal.SIGTERM, sig_handler)
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", "--err", required=False, type=str, default="mae", metavar="<error metric>", help="Error metric for determining the model with minimum error value")
        args = parser.parse_args()
        DEFAULT_ERROR_KEYS = args.err.split(",")
        server = EstimatorServer(SERVE_SOCKET)
        server.start()
    finally:
        print("estimator exit")
        clean_socket()


if __name__ == "__main__":
    run()
