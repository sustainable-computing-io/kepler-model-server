import requests

import os
import sys
import shutil

import codecs

server_path = os.path.join(os.path.dirname(__file__), '../server')
util_path = os.path.join(os.path.dirname(__file__), '../server/util')
train_path = os.path.join(os.path.dirname(__file__), '../server/train')

sys.path.append(server_path)
sys.path.append(util_path)
sys.path.append(train_path)

from train.train_types import FeatureGroup, FeatureGroups, ModelOutputType, is_weight_output
from model_server import MODEL_SERVER_PORT

def get_model_request_json(metrics, output_type):
    return {"metrics": metrics, "output_type": output_type.name}

TMP_FILE = 'download.zip'
DOWNLOAD_FOLDER = 'download'
def make_request(metrics, output_type):
    model_request = get_model_request_json(metrics, output_type)
    response = requests.post('http://localhost:{}/model'.format(MODEL_SERVER_PORT), json=model_request)
    assert response.status_code == 200, response.text
    if is_weight_output(output_type):
        print(response.text)
    else:
        output_path = os.path.join(DOWNLOAD_FOLDER, output_type.name)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        with codecs.open(TMP_FILE, 'wb') as f:
            f.write(response.content)
        shutil.unpack_archive(TMP_FILE, output_path)
        os.remove(TMP_FILE)
        
if __name__ == '__main__':
    source = os.path.join(os.path.dirname(__file__), "test_models")
    destination = os.path.join(os.path.dirname(__file__), "../server/models")
    # !! remove the existing local path
    if os.path.exists(destination):
        shutil.rmtree(destination)
    # copy test_models to local model path
    shutil.copytree(source, destination)
    # full features
    metrics = FeatureGroups[FeatureGroup.Full]
    # archived model
    output_type = ModelOutputType.AbsPower
    make_request(metrics, output_type)
    # model component weight
    output_type = ModelOutputType.AbsComponentModelWeight
    make_request(metrics, output_type)
    # # model weight #TODO
    # output_type = ModelOutputType.AbsModelWeight
    # make_request(metrics, output_type)