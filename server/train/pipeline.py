import os
import sys
import shutil

server_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(server_path)

from abc import ABCMeta, abstractmethod
from train_types import FeatureGroup, get_feature_group, ModelOutputType, is_weight_output, POWER_COMPONENTS
import json

from util.config import getConfig, getPath
from util.loader import get_save_path, get_archived_file, download_and_save

METADATA_FILENAME = 'metadata.json'

model_path =  getPath(getConfig('MODEL_PATH', 'models'))

initial_models_location = getConfig('INITIAL_MODELS_LOC', None)
if initial_models_location is not None:
    initial_model_names = {m.split('=')[0]: m.split('=')[1] for m in  getConfig('INITIAL_MODEL_NAMES', "").split('\n') if m != ''}
else:
    initial_model_names = dict()

def get_model_group_path(output_type, feature_group):
    return os.path.join(model_path, output_type.name, feature_group.name)

def load_inital_model(output_type, feature_group):
    output_type_name = output_type.name
    feature_group_name = feature_group.name
    if output_type_name not in initial_model_names:
        return
    model_name = initial_model_names[output_type_name]
    # prepare directory
    group_path = get_model_group_path(output_type, feature_group)
    save_path = get_save_path(group_path, model_name) 
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    metadata_url = os.path.join(initial_models_location, output_type_name, feature_group_name, model_name, METADATA_FILENAME)
    metadata_file = os.path.join(save_path, METADATA_FILENAME)
    metadata_file = download_and_save(metadata_url, metadata_file)
    if metadata_file is not None:
        # weight-type
        if output_type == ModelOutputType.AbsComponentModelWeight or output_type == ModelOutputType.AbsModelWeight \
            or output_type == ModelOutputType.DynComponentModelWeight or output_type == ModelOutputType.DynModelWeight:
            model_filename = model_name + ".json"
            model_file = os.path.join(save_path, model_filename)
            model_url = os.path.join(initial_models_location, output_type_name, feature_group_name, model_name, model_file)
        else:
            model_file = save_path + ".zip"
            model_url = os.path.join(initial_models_location, output_type_name, feature_group_name, model_name  + ".zip")
        download_and_save(model_url, model_file)

for ot in ModelOutputType:
    ot_group_path = os.path.join(model_path, ot.name)
    if not os.path.exists(ot_group_path):
        os.mkdir(ot_group_path)
    for g in FeatureGroup:
        group_path = os.path.join(ot_group_path, g.name)
        if not os.path.exists(group_path):
            os.mkdir(group_path)
        load_inital_model(ot, g)

class TrainPipeline(metaclass=ABCMeta):
    def __init__(self, model_name, model_class, model_file, features, output_type):
        self.model_name = model_name
        self.model_class = model_class
        self.model_file = model_file
        self.features = features
        self.output_type = output_type
        self.feature_group = get_feature_group(features)
        self.weight_type = is_weight_output(self.output_type)
        self.group_path = get_model_group_path(self.output_type, self.feature_group)
        print(self.group_path)
        self.save_path = get_save_path(self.group_path, self.model_name) 
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    # call this only model is updated
    def update_metadata(self, item):
        print('update metadata')
        item['model_name'] = self.model_name
        item['model_class'] = self.model_class
        item['model_file'] = self.model_file
        item['features']= self.features
        item['fe_files'] = [] if not hasattr(self, 'fe_files') else self.fe_files
        item['output_type'] = self.output_type.name
        self.metadata = item
        metadata_file = os.path.join(self.save_path, METADATA_FILENAME)
        with open(metadata_file, "w") as f:
            json.dump(item, f)
        if not self.weight_type:
            self.archive_model()
            
    @abstractmethod
    def train(self, prom_client):
        return NotImplemented

    def archive_model(self):
        archived_file = get_archived_file(self.group_path, self.model_name) 
        print("achive model ", archived_file, self.save_path)
        shutil.make_archive(self.save_path, 'zip', self.save_path)

    def save_comp_model(self, model_file_dict, model_features_dict, model_fe_file_dicts={}):
        model_dict = dict()
        for comp in POWER_COMPONENTS:
            model_dict[comp] = {
                'model_file': model_file_dict[comp],
                'features': model_features_dict[comp],
                'fe_files': [] if comp not in model_fe_file_dicts else model_fe_file_dicts[comp]  # optional
            }
        model_file = os.path.join(self.save_path, self.model_file)
        with open(model_file, "w") as f:
            json.dump(model_dict, f)