import json
import logging
import os
import shutil
import signal
import socket
import sys

import click
import pandas as pd

from kepler_model.estimate.archived_model import get_achived_model
from kepler_model.estimate.model.model import load_downloaded_model
from kepler_model.estimate.model_server_connector import is_model_server_enabled, make_request
from kepler_model.train.profiler.node_type_index import NodeTypeSpec, discover_spec_values, get_machine_spec
from kepler_model.util.config import CONFIG_PATH, SERVE_SOCKET, download_path, set_env_from_model_config, set_config_dir
from kepler_model.util.loader import get_download_output_path, load_metadata
from kepler_model.util.train_types import ModelOutputType, convert_enery_source, is_output_type_supported

###############################################
# power request

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


def handle_request(data: str, machine_spec=None, discovered_core=None) -> dict:
    try:
        power_request = json.loads(data, object_hook=lambda d: PowerRequest(**d))
    except Exception as e:
        msg = f"failed to handle request: {e}"
        logger.error(msg)
        return {"powers": dict(), "msg": msg}

    if not is_output_type_supported(power_request.output_type):
        msg = f"output type {power_request.output_type} is not supported"
        logger.error(msg)
        return {"powers": dict(), "msg": msg}

    output_type = ModelOutputType[power_request.output_type]
    power_request.energy_source = convert_enery_source(power_request.energy_source)

    if output_type.name not in loaded_model:
        loaded_model[output_type.name] = dict()

    output_path = ""
    mismatch_trainer = False
    if is_model_server_enabled():
        if power_request.trainer_name is not None and power_request.trainer_name:
            if output_type.name in loaded_model and power_request.energy_source in loaded_model[output_type.name]:
                current_trainer = loaded_model[output_type.name][power_request.energy_source].trainer_name
                mismatch_trainer = current_trainer != power_request.trainer_name
                if mismatch_trainer:
                    logger.info(f"try obtaining the requesting trainer {power_request.trainer_name} (current: {current_trainer})")
    if power_request.energy_source not in loaded_model[output_type.name] or mismatch_trainer:
        output_path = get_download_output_path(download_path, power_request.energy_source, output_type)
        if mismatch_trainer and os.path.exists(output_path):
            # remove existing model if mismatch
            shutil.rmtree(output_path)
        if not os.path.exists(output_path):
            # try connecting to model server
            output_path = make_request(power_request, machine_spec)
            if output_path is None:
                # find from config
                output_path = get_achived_model(power_request)
                if output_path is None:
                    msg = f"failed to get model from request {data}"
                    logger.error(msg)
                    return {"powers": dict(), "msg": msg}
                logger.info(f"load model from config: {output_path}")
            else:
                logger.info(f"load model from model server: {output_path}")

        loaded_item = load_downloaded_model(power_request.energy_source, output_type)

        if loaded_item is not None and loaded_item.estimator is not None:
            loaded_model[output_type.name][power_request.energy_source] = loaded_item
            logger.info(f"set model {loaded_item.model_name} for {output_type.name} ({power_request.energy_source})")
        else:
            msg = f"load item for {power_request.energy_source} is none"
            logger.error(msg)
            return {"powers": dict(), "msg": msg}

    model = loaded_model[output_type.name][power_request.energy_source]
    powers, msg = model.get_power(power_request.datapoint)
    if msg != "":
        logger.info(f"{model.model_name} failed to predict; removed: {msg}")
        if output_path != "" and os.path.exists(output_path):
            shutil.rmtree(output_path)
    response = {"powers": powers, "msg": msg}
    # add core_ratio if applicable
    core_ratio = 1
    if discovered_core is not None and discovered_core > 0:
        metadata = load_metadata(output_path)
        if metadata is not None and "machine_spec" in metadata:
            model_spec = NodeTypeSpec(**metadata["machine_spec"])
            model_cores = model_spec.get_cores()
            if model_cores > 0:
                core_ratio = discovered_core / model_cores
            logger.debug(f"model cores: {model_cores}")
        logger.debug(f"metadata: {metadata}")
    response["core_ratio"] = core_ratio

    return response


class EstimatorServer:
    def __init__(self, socket_path, machine_spec):
        self.socket_path = socket_path
        self.machine_spec = machine_spec
        spec_values = discover_spec_values()
        discovered_spec = NodeTypeSpec(**spec_values)
        self.discovered_core = discovered_spec.get_cores()
        logger.info(f"initialize EstimatorServer with spec={machine_spec}, discovered_core={self.discovered_core}")

    def start(self):
        s = self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(self.socket_path)
        s.listen(1)
        logger.info(f"listening on {self.socket_path}")
        try:
            while True:
                connection, _ = s.accept()
                self.accepted(connection)
        finally:
            try:
                os.remove(self.socket_path)
                sys.stdout.write("close socket\n")
            except Exception as e:
                logger.error(f"failed to close socket: {e}")

    def accepted(self, connection):
        data = b""
        while True:
            shunk = connection.recv(1024).strip()
            data += shunk
            if shunk is None or shunk.decode()[-1] == "}":
                break
        decoded_data = data.decode()
        y = handle_request(decoded_data, self.machine_spec, self.discovered_core)
        response = json.dumps(y)
        connection.send(response.encode())


def clean_socket():
    logger.info("clean socket")
    if os.path.exists(SERVE_SOCKET):
        os.unlink(SERVE_SOCKET)


def sig_handler(signum, frame) -> None:
    clean_socket()
    sys.exit(1)


@click.command()
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["debug", "info", "warn", "error"]),
    default="info",
    required=False,
)
@click.option(
    "--machine-spec",
    type=click.Path(exists=True),
    required=False,
)
@click.option(
    "--config-dir",
    "-c",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    default=CONFIG_PATH,
    required=False,
)
def run(log_level: str, machine_spec: str, config_dir: str) -> int:
    level = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("starting estimator")
    set_config_dir(config_dir)

    set_env_from_model_config()
    clean_socket()
    signal.signal(signal.SIGTERM, sig_handler)
    try:
        spec = get_machine_spec(machine_spec)
        server = EstimatorServer(SERVE_SOCKET, spec)
        server.start()
    finally:
        click.echo("estimator exit")
        clean_socket()

    return 0


if __name__ == "__main__":
    sys.exit(run())
