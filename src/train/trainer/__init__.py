import os
import sys
import shutil
import json
import pickle

src_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(src_path)

from abc import ABCMeta, abstractmethod

from train import PowerSourceMap, ModelOutputType, FeatureGroup, FeatureGroups
from train import is_weight_output, node_info_column, component_to_col

from util.loader import model_path, get_model_group_path, get_save_path, download_and_save, get_model_name, get_archived_file, METADATA_FILENAME, SCALER_FILENAME, WEIGHT_FILENAME, CHECKPOINT_FOLDERNAME
from util.config import getConfig

default_initial_models_location = "https://raw.githubusercontent.com/sustainable-computing-io/kepler-model-db/main/models"
initial_models_location = getConfig('INITIAL_MODELS_LOC', default_initial_models_location)

def get_checkpoint_path(group_path):
    return os.path.join(group_path, CHECKPOINT_FOLDERNAME)

for energy_source in PowerSourceMap.keys():
    src_group_path = os.path.join(model_path, energy_source)
    if not os.path.exists(src_group_path):
        os.mkdir(src_group_path)
    for ot in ModelOutputType:
        ot_group_path = os.path.join(src_group_path, ot.name)
        if not os.path.exists(ot_group_path):
            os.mkdir(ot_group_path)
        for g in FeatureGroup:
            if g == FeatureGroup.Unknown:
                continue
            group_path = os.path.join(ot_group_path, g.name)
            if not os.path.exists(group_path):
                os.mkdir(group_path)
            checkpoint_path = get_checkpoint_path(group_path)
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_and_split(X_values, y_values, scaler, test_size=0.1):
    features = scaler.transform(X_values)
    X_train, X_test, y_train, y_test = train_test_split(features, y_values, test_size=test_size, shuffle=True)
    return X_train, X_test, y_train, y_test


class Trainer(metaclass=ABCMeta):
    def __init__(self, profiles, model_class, energy_components, feature_group, energy_source, node_level, scaler_type="minmax"):
        self.profiles = profiles
        self.energy_components = energy_components
        self.feature_group_name = feature_group
        self.feature_group = FeatureGroup[feature_group]
        self.features = FeatureGroups[self.feature_group]
        self.energy_source = energy_source
        self.node_level = node_level

        self.trainer_name = self.__class__.__name__
        self.model_class = model_class
        self.output_type = ModelOutputType.AbsPower if node_level else ModelOutputType.DynPower
        self.group_path = get_model_group_path(self.output_type, self.feature_group, self.energy_source)
        self.checkpoint_toppath = get_checkpoint_path(self.group_path)
        self.node_models = dict()
        self.node_scalers = dict()
        self.scaler_type = scaler_type

    def _get_save_path(self, node_type):
        save_path = get_save_path(self.group_path, self.trainer_name, node_type=node_type) 
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        return save_path

    def _model_filename(self, node_type):
        model_name = get_model_name(self.trainer_name, node_type)
        model_file = model_name + ".json"
        return model_name, model_file

    def _checkpoint_filename(self, component, node_type):
        return "{}_{}_{}".format(self.trainer_name, component, node_type)

    def _checkpoint_filepath(self, component, node_type):
        checkpoint_filename = self._checkpoint_filename(component, node_type)
        return os.path.join(self.checkpoint_toppath, checkpoint_filename)

    def _remote_checkpoint(self, component, node_type):
        output_type_name = self.output_type.name
        feature_group_name = self.feature_group.name
        checkpoint_filename = self._checkpoint_filename(component, node_type)
        return os.path.join(initial_models_location, output_type_name, feature_group_name, checkpoint_filename)
        
    @abstractmethod
    def init_model(self):
        return NotImplemented

    @abstractmethod
    def train(self, node_type, component, X_values, y_values):
        return NotImplemented
        
    @abstractmethod
    def save_checkpoint(self, model, filepath):
        return NotImplemented

    @abstractmethod
    def load_local_checkpoint(self, filepath):
        return NotImplemented

    @abstractmethod
    def load_remote_checkpoint(self, url):
        return NotImplemented

    @abstractmethod
    def should_archive(self, node_type):
        return NotImplemented

    @abstractmethod
    def get_basic_metadata(self, node_type):
        return NotImplemented

    @abstractmethod
    def save_model(self, component_save_path, node_type, component):
        return NotImplemented

    @abstractmethod
    def component_model_filename(self, component):
        return NotImplemented

    @abstractmethod
    def get_mae(self, node_type, component, X_test, y_test):
        return NotImplemented

    @abstractmethod
    def get_weight_dict(self, node_type):
        return NotImplemented

    def load_model(self, node_type):
        # set model
        if node_type not in self.node_models:
            self.node_models[node_type] = dict()
        for component in self.energy_components:
            # try loading checkpoint
            local_checkpoint = self._checkpoint_filepath(component, node_type)
            model, ok = self.load_local_checkpoint(local_checkpoint)
            if not ok:
                url_path = self._remote_checkpoint(component, node_type)
                model, ok = self.load_remote_checkpoint(url_path)
                if ok:
                    self.print_log("Load initial checkpoint from {}".format(url_path))
            if ok:
                self.node_models[node_type][component] = model
                self.print_log("Continue from last checkpoint ({})".format(component))
            else:
                # init if failed to load any checkpoint
                self.node_models[node_type][component] = self.init_model()
                self.print_log("Newly initialize model ({})".format(component))

    def load_scaler(self, node_type):
        if node_type not in self.profiles:
            return None
        if self.scaler_type == "standard":
            return self.profiles[node_type].get_standard_scaler(self.feature_group_name)
        else:
            return self.profiles[node_type].get_minmax_scaler(self.feature_group_name)

    def process(self, data, power_labels):
        node_types = pd.unique(data[node_info_column])
        for node_type in node_types:
            self.load_model(node_type)
            scaler = self.load_scaler(node_type)
            node_type_filtered_data = data[data[node_info_column] == node_type]
            if scaler is None:
                # no profiled scaler
                x_values = node_type_filtered_data[self.features].values
                if self.scaler_type == "standard":
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                scaler.fit(x_values)
                self.print_log("Cannot load scaler for {}/{}, fit scaler to latest data".format(node_type, self.feature_group_name))
            self.node_scalers[node_type] = scaler
            
            for component in self.energy_components:
                power_label = component_to_col(component) 
                related_labels = [label for label in power_labels if power_label in label]
                X_values = node_type_filtered_data[self.features].values
                y_values = node_type_filtered_data[related_labels].mean(axis=1)
                X_train, X_test, y_train, y_test = normalize_and_split(X_values, y_values, scaler=self.node_scalers[node_type])
                self.train(node_type, component, X_train, y_train)
                self.save_checkpoint(self.node_models[node_type][component], self._checkpoint_filepath(component, node_type))
            if self.should_archive(node_type):
                self.save_model_and_metadata(node_type, X_test, y_test)

    def save_metadata(self, node_type, mae, item):
        save_path = self._get_save_path(node_type)
        model_name, model_file = self._model_filename(node_type)
        item['model_name'] = model_name
        item['model_class'] = self.model_class
        item['model_file'] = model_file
        item['features']= self.features
        item['fe_files'] = [] if not hasattr(self, 'fe_files') else self.fe_files
        item['output_type'] = self.output_type.name
        item['mae'] = mae
        self.metadata = item
        metadata_file = os.path.join(save_path, METADATA_FILENAME)
        with open(metadata_file, "w") as f:
            json.dump(item, f)

    def archive_model(self, node_type):
        save_path = self._get_save_path(node_type)
        model_name, _ = self._model_filename(node_type)
        archived_file = get_archived_file(self.group_path, model_name) 
        print("archive model ", archived_file)
        shutil.make_archive(save_path, 'zip', save_path)
        weight_dict = self.get_weight_dict(node_type)
        if weight_dict is not None:
            weight_file = os.path.join(save_path, WEIGHT_FILENAME)
            with open(weight_file, "w") as f:
                json.dump(weight_dict, f)

    def save_scaler(self, save_path, node_type):
        filename = os.path.join(save_path, SCALER_FILENAME)
        with open(filename, "wb") as f:
            pickle.dump(self.node_scalers[node_type], f)

    def save_model_and_metadata(self, node_type, X_test, y_test):
        save_path = self._get_save_path(node_type)
        _, model_file = self._model_filename(node_type)
        model_dict = dict()
        for component in self.energy_components:
            component_save_file = self.component_model_filename(component)
            model_dict[component] = {
                'model_file': component_save_file,
                'features': self.features,
                'fe_files': [SCALER_FILENAME]
            }
            # save component model
            self.save_model(save_path, node_type, component)
        self.save_scaler(save_path, node_type)
        # save model dict
        model_filepath = os.path.join(save_path, model_file)
        with open(model_filepath, "w") as f:
            json.dump(model_dict, f)
        self.archive_model(node_type)
        # save metadata
        max_mae = None
        item = self.get_basic_metadata(node_type)
        for component in self.energy_components:
            mae = self.get_mae(node_type, component, X_test, y_test)
            if max_mae is None or mae > max_mae:
                max_mae = mae
        self.save_metadata(node_type, max_mae, item)

    def predict(self, node_type, component, X_values):
        features = self.node_scalers[node_type].transform(X_values)
        model = self.node_models[node_type][component]
        return model.predict(features)

    def print_log(self, message):
        print("{} trainer ({}/{}/{}): {}".format(self.trainer_name, "Abs" if self.node_level else "Dyn", self.feature_group, self.energy_source, message))

    def get_metadata(self):
        items = []
        for node_type in self.node_models.keys():
            save_path = self._get_save_path(node_type)
            metadata_file = os.path.join(save_path, METADATA_FILENAME)
            with open(metadata_file, "r") as f:
                item = json.load(f)
                item['node_type'] = node_type
                items += [item]
        return pd.DataFrame(items)
        
                
