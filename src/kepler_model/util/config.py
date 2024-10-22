#################################################
# config.py
#
# getConfig: return value set by configuration
#            which can be from config map or environment variable
#            if not provided, return default value
# getPath:   return path relative to mount path
#            create new if not exists
#            mount path is set by configuration
#            if mount path cannot be write,
#            set to local folder (/server)
#
#################################################

import logging
import os
import pathlib
import typing

import requests

from .loader import base_model_url, default_init_model_name, default_pipelines, default_train_output_pipeline, get_pipeline_url, get_url
from .train_types import FeatureGroup, ModelOutputType, is_output_type_supported

logger = logging.getLogger(__name__)


# can be read only (for configmap mount)
CONFIG_PATH = pathlib.Path(os.environ.get("CONFIG_PATH", "/etc/kepler/kepler.config"))

DEFAULT_TOTAL_SOURCE = "acpi"
DEFAULT_COMPONENTS_SOURCE = "rapl-sysfs"
TOTAL_KEY = "TOTAL"
COMPONENTS_KEY = "COMPONENTS"


MODEL_SERVER_SVC = "kepler-model-server.kepler.svc.cluster.local"
DEFAULT_MODEL_SERVER_PORT = 8100
MODEL_SERVER_ENDPOINT = f"http://{MODEL_SERVER_SVC}:{DEFAULT_MODEL_SERVER_PORT}"
MODEL_SERVER_MODEL_REQ_PATH = "/model"
MODEL_SERVER_MODEL_LIST_PATH = "/best-models"

SERVE_SOCKET = "/tmp/estimator.sock"


def set_config_dir(config_dir: pathlib.Path):
    global CONFIG_PATH
    CONFIG_PATH = config_dir


T_conf = typing.TypeVar("T_conf", int, float, bool, str, pathlib.Path, list[str])


def _value_or_default(value: str | None, default: T_conf) -> T_conf:
    if value == "" or value is None:
        return default

    if isinstance(default, bool):
        return value == "true"

    if isinstance(default, int):
        return int(value)

    if isinstance(default, float):
        return float(value)

    if isinstance(default, str):
        return value

    if isinstance(default, pathlib.Path):
        return pathlib.Path(value)

    if isinstance(default, list):
        return [v.strip() for v in value.split(",")]


def get_config(key: str, default: T_conf) -> T_conf:
    # check configmap path
    file = CONFIG_PATH / key

    # CONFIG_FILES take precedence over env
    if os.path.exists(file):
        with open(file) as f:
            return _value_or_default(f.read().strip(), default)

    # check env
    if key in os.environ:
        return _value_or_default(os.environ[key].strip(), default)

    return default


def _init_mnt_path() -> pathlib.Path:
    # update value from environment if exists
    # must be writable (for shared volume mount)
    mnt_path = get_config("MNT_PATH", pathlib.Path("/mnt"))

    if not os.path.exists(mnt_path) or not os.access(mnt_path, os.W_OK):
        # use local path if not exists or cannot write
        tmp_mnt_path = pathlib.Path("/tmp/model-server/mnt")
        logger.warning(f"cannot write to MNT_PATH - {mnt_path}, falling back to using tmp path: {tmp_mnt_path}")
        mnt_path = tmp_mnt_path

    os.makedirs(mnt_path, exist_ok=True)
    return mnt_path


MNT_PATH = _init_mnt_path()


model_topurl = get_config("MODEL_TOPURL", base_model_url)


def _init_initial_pipeline_urls(model_topurl) -> list[str]:
    urls = get_config("INITIAL_PIPELINE_URL", [])
    if urls:
        return urls

    if model_topurl == base_model_url:
        return [get_pipeline_url(model_topurl=model_topurl, pipeline_name=p) for p in default_pipelines.values()]

    return [get_pipeline_url(model_topurl=model_topurl, pipeline_name=default_train_output_pipeline)]


initial_pipeline_urls = _init_initial_pipeline_urls(model_topurl)

model_toppath = get_config("MODEL_PATH", MNT_PATH / "models")
download_path = get_config("MODEL_PATH", MNT_PATH / "download")
os.makedirs(download_path, exist_ok=True)
os.makedirs(model_toppath, exist_ok=True)

ERROR_KEY = get_config("ERROR_KEY", "mae")

RESOURCE_DIR = get_config("RESOURCE_DIR", pathlib.Path("/tmp/model-server/resources"))
os.makedirs(RESOURCE_DIR, exist_ok=True)


MODEL_SERVER_ENABLE = get_config("MODEL_SERVER_ENABLE", False)


def is_model_server_enabled() -> bool:
    MODEL_SERVER_ENABLE = get_config("MODEL_SERVER_ENABLE", False)
    return MODEL_SERVER_ENABLE


def _model_server_endpoint():
    MODEL_SERVER_URL = get_config("MODEL_SERVER_URL", MODEL_SERVER_SVC)
    if MODEL_SERVER_URL == MODEL_SERVER_SVC:
        MODEL_SERVER_PORT = get_config("MODEL_SERVER_PORT", DEFAULT_MODEL_SERVER_PORT)
        modelServerEndpoint = f"http://{MODEL_SERVER_URL}:{MODEL_SERVER_PORT}"
    else:
        modelServerEndpoint = MODEL_SERVER_URL
    return modelServerEndpoint


def get_model_server_req_endpoint():
    return _model_server_endpoint() + get_config("MODEL_SERVER_MODEL_REQ_PATH", MODEL_SERVER_MODEL_REQ_PATH)


def get_model_server_list_endpoint():
    return _model_server_endpoint() + get_config("MODEL_SERVER_MODEL_LIST_PATH", MODEL_SERVER_MODEL_LIST_PATH)


# set_env_from_model_config: extract environment values based on environment key MODEL_CONFIG
def set_env_from_model_config():
    model_config = get_config("MODEL_CONFIG", "")
    if not model_config:
        return

    for line in model_config.splitlines():
        line = line.strip()
        # ignore comments and blanks
        if not line or line.startswith("#"):
            continue

        # pick only the first part until # and ignore the rest
        splits = line.split("#")[0].strip().split("=")
        if len(splits) > 1:
            os.environ[splits[0].strip()] = splits[1].strip()
            logging.info(f"set env {splits[0]} to '{splits[1]}'.")


def get_init_url(prefix):
    envKey = "_".join([prefix, "INIT_URL"])
    return get_config(envKey, "")


def get_energy_source(prefix):
    if TOTAL_KEY in prefix:
        return DEFAULT_TOTAL_SOURCE
    if COMPONENTS_KEY in prefix:
        return DEFAULT_COMPONENTS_SOURCE


# get_init_model_url: get initial model from URL if estimator is enabled
def get_init_model_url(energy_source, output_type, model_topurl=model_topurl):
    if base_model_url == model_topurl:
        pipeline_name = default_pipelines[energy_source]
    else:
        pipeline_name = default_train_output_pipeline

    model_prefixes = ["_".join([level, coverage]) for level in ["NODE", "CONTAINER", "PROCESS"] for coverage in [TOTAL_KEY, COMPONENTS_KEY]]
    for prefix in model_prefixes:
        if get_energy_source(prefix) == energy_source:
            modelURL = get_init_url(prefix)
            logger.info(f"get init url: {modelURL}")
            url = get_url(
                feature_group=FeatureGroup.BPFOnly,
                output_type=ModelOutputType[output_type],
                energy_source=energy_source,
                model_topurl=model_topurl,
                pipeline_name=pipeline_name,
            )
            if modelURL == "" and is_output_type_supported(output_type):
                if energy_source in default_init_model_name:
                    model_name = default_init_model_name[energy_source]
                    modelURL = get_url(
                        feature_group=FeatureGroup.BPFOnly,
                        output_type=ModelOutputType[output_type],
                        energy_source=energy_source,
                        model_topurl=model_topurl,
                        pipeline_name=pipeline_name,
                        model_name=model_name,
                    )
                    if url:
                        response = requests.get(url)
                        if response.status_code == 200:
                            modelURL = url
                logger.warn(f"init URL is not set, using {modelURL}")
            return modelURL
    logger.warn(f"no matching config for {output_type}, {energy_source} found")
    return ""
