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

# must be writable (for shared volume mount)
MNT_PATH = "/mnt"
# can be read only (for configmap mount)
CONFIG_PATH = "/etc/kepler/kepler.config"

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
print("mount path: ", MNT_PATH)

CONFIG_PATH = getConfig('CONFIG_PATH', CONFIG_PATH)