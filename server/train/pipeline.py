import os
import sys

server_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(server_path)

from abc import ABCMeta, abstractmethod
from train_types import FeatureGroup, get_feature_group
import json

from util.config import getConfig, getPath

METADATA_FILENAME = 'metadata.json'

model_path =  getPath(getConfig('MODEL_PATH', 'models'))

for g in FeatureGroup:
    group_path = os.path.join(model_path, g.name)
    if not os.path.exists(group_path):
        os.mkdir(group_path)

class TrainPipeline(metaclass=ABCMeta):
    def __init__(self, model_name, model_class, model_file, features):
        self.model_name = model_name
        self.model_class = model_class
        self.model_file = model_file
        self.features = features
        self.model_group = get_feature_group(features)
        group_path = os.path.join(model_path, self.model_group.name)
        print(group_path)
        self.save_path = os.path.join(group_path, self.model_name)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def update_metadata(self, item):
        print('update metadata')
        item['model_name'] = self.model_name
        item['model_class'] = self.model_class
        item['model_file'] = self.model_file
        item['features']= self.features
        item['fe_files'] = [] if not hasattr(self, 'fe_files') else self.fe_files
        self.metadata = item
        metadata_file = os.path.join(self.save_path, METADATA_FILENAME)
        with open(metadata_file, "w") as f:
            json.dump(item, f)
        
    @abstractmethod
    def train(self, prom_client):
        return NotImplemented

    def get_model_specific_metadata(self):
        return dict()
