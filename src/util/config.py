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

import os
from loader import get_url, get_pipeline_url
from train_types import ModelOutputType, is_support_output_type

# must be writable (for shared volume mount)
MNT_PATH = "/mnt"
# can be read only (for configmap mount)
CONFIG_PATH = "/etc/kepler/kepler.config"

DOWNLOAD_FOLDERNAME = "download"
MODEL_FOLDERNAME = "models"

DEFAULT_TOTAL_SOURCE = "acpi"
DEFAULT_COMPONENTS_SOURCE = "intel_rapl"
TOTAL_KEY = "TOTAL"
COMPONENTS_KEY = "COMPONENTS"

modelConfigPrefix = [ "_".join([level, coverage]) for level in ["NODE", "CONTAINER", "PROCESS"] for coverage in [TOTAL_KEY, COMPONENTS_KEY]]

MODEL_SERVER_SVC = "kepler-model-server.kepler.svc.cluster.local"
DEFAULT_MODEL_SERVER_PORT = 8100
MODEL_SERVER_ENDPOINT = 'http://{}:{}'.format(MODEL_SERVER_SVC, DEFAULT_MODEL_SERVER_PORT)
MODEL_SERVER_MODEL_REQ_PATH = "/model"
MODEL_SERVER_MODEL_LIST_PATH = "/best-models"
MODEL_SERVER_ENABLE = False

SERVE_SOCKET = '/tmp/estimator.sock'

def getConfig(key, default):
    # check configmap path
    file = os.path.join(CONFIG_PATH, key)
    if os.path.exists(file):
        with open(file, "r") as f:
            return f.read()
    # check env
    return os.getenv(key, default)

def getPath(subpath):
    path = os.path.join(MNT_PATH, subpath)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

# update value from environment if exists
MNT_PATH = getConfig('MNT_PATH', MNT_PATH)
if not os.path.exists(MNT_PATH) or not os.access(MNT_PATH, os.W_OK):
    # use local path if not exists or cannot write
    MNT_PATH = os.path.join(os.path.dirname(__file__), '..')

CONFIG_PATH = getConfig('CONFIG_PATH', CONFIG_PATH)

initial_pipeline_url = getConfig('INITIAL_PIPELINE_URL', get_pipeline_url())

model_toppath =  getConfig('MODEL_PATH', getPath(MODEL_FOLDERNAME))
download_path = getConfig('MODEL_PATH', getPath(DOWNLOAD_FOLDERNAME))

ERROR_KEY = 'mae'
ERROR_KEY = getConfig('ERROR_KEY', ERROR_KEY)

def is_model_server_enabled():
    return getConfig('MODEL_SERVER_ENABLE', "false").lower() == "true"

def _model_server_endpoint():
    MODEL_SERVER_URL = getConfig('MODEL_SERVER_URL', MODEL_SERVER_SVC)
    if MODEL_SERVER_URL == MODEL_SERVER_SVC:
        MODEL_SERVER_PORT = getConfig('MODEL_SERVER_PORT', DEFAULT_MODEL_SERVER_PORT)
        MODEL_SERVER_PORT = int(MODEL_SERVER_PORT)
        modelServerEndpoint = 'http://{}:{}'.format(MODEL_SERVER_URL, MODEL_SERVER_PORT)
    else:
        modelServerEndpoint = MODEL_SERVER_URL
    return modelServerEndpoint

def get_model_server_req_endpoint():
    return _model_server_endpoint() + getConfig('MODEL_SERVER_MODEL_REQ_PATH', MODEL_SERVER_MODEL_REQ_PATH)

def get_model_server_list_endpoint():
    return _model_server_endpoint() + getConfig('MODEL_SERVER_MODEL_LIST_PATH', MODEL_SERVER_MODEL_LIST_PATH)

# set_env_from_model_config: extract environment values based on environment key MODEL_CONFIG
def set_env_from_model_config():
    model_config = getConfig('MODEL_CONFIG', "")
    if model_config != "":
        lines = model_config.splitlines()
        for line in lines:
            splits = line.split('=')
            if len(splits) > 1:
                os.environ[splits[0]] = splits[1]
                print("set {} to {}.".format(splits[0], splits[1]))


def is_estimator_enable(prefix):
    envKey = "_".join([prefix, "ESTIMATOR"])
    value = getConfig(envKey, "")
    return value.lower() == "true"

def get_init_url(prefix):
    envKey = "_".join([prefix, "INIT_URL"])
    return getConfig(envKey, "")

def get_energy_source(prefix):
    if TOTAL_KEY in prefix:
        return DEFAULT_TOTAL_SOURCE
    if COMPONENTS_KEY in prefix:
        return DEFAULT_COMPONENTS_SOURCE

# get_init_model_url: get initial model from URL if estimator is enabled
def get_init_model_url(energy_source, output_type):
    for prefix in modelConfigPrefix:
        if get_energy_source(prefix) == energy_source:
            modelURL = get_init_url(prefix)
            print("get init url", modelURL)
            if modelURL == "" and is_support_output_type(output_type):
                print("init URL is not set, try using default URL".format(output_type))
                return get_url(output_type=ModelOutputType[output_type], energy_source=energy_source)
            else:
                return modelURL
    print("no match config for {}, {}".format(output_type, energy_source))
    return ""