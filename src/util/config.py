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


modelItemNameMap = {
    # "Abs": "NODE_POWER_MODEL",
    # "Dyn": "CONTAINER_POWER_MODEL"
    "Abs": "NODE_TOTAL",
    "Dyn": "CONTAINER_TOTAL"
}

estimatorKeyMap = {"{}Power".format(modelItemKey): "{}_ESTIMATOR".format(modelItemNameMap[modelItemKey]) for modelItemKey in modelItemNameMap.keys()}
initUrlKeyMap = {"{}Power".format(modelItemKey): "{}_INIT_URL".format(modelItemNameMap[modelItemKey]) for modelItemKey in modelItemNameMap.keys()}

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

default_model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
model_toppath =  getPath(getConfig('MODEL_PATH', default_model_path))

if not os.path.exists(model_toppath):
    os.mkdir(model_toppath)

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


# get_init_model: get initial model from URL if estimator is enabled
def get_init_model(output_type):
    print("get_init_model: {}", initUrlKeyMap, estimatorKeyMap, output_type)
    if output_type in estimatorKeyMap:
        enabled = getConfig(estimatorKeyMap[output_type], "true").lower() == "true"
        if enabled:
            modelURL = getConfig(initUrlKeyMap[output_type], "")
            if modelURL == "" and is_support_output_type(output_type):
                print("get default init URL, {} is not set", initUrlKeyMap[output_type])
                return get_url(output_type=ModelOutputType[output_type])
            else:
                return modelURL
        else:
            print("{} is not enaled".format(output_type))
    return ""

