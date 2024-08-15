import shutil

from abc import ABCMeta, abstractmethod

import os


from kepler_model.util import assure_path, ModelOutputType, FeatureGroups, FeatureGroup, save_json, save_metadata, load_metadata, save_scaler, save_weight

from kepler_model.util.prom_types import node_info_column
from kepler_model.util.train_types import main_feature
from kepler_model.util.extract_types import component_to_col, get_unit_vals, ratio_to_col
from kepler_model.util.loader import get_model_group_path, get_save_path, get_model_name, get_archived_file, CHECKPOINT_FOLDERNAME, load_scaler
from kepler_model.util.config import model_toppath


def get_assured_checkpoint_path(group_path, assure=True):
    checkpoint_path = os.path.join(group_path, CHECKPOINT_FOLDERNAME)
    if assure:
        assure_path(checkpoint_path)
    return checkpoint_path


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler


def normalize_and_split(X_values, y_values, scaler, test_size=0.1):
    features = scaler.transform(X_values)
    X_train, X_test, y_train, y_test = train_test_split(features, y_values, test_size=test_size, shuffle=True)
    return X_train, X_test, y_train, y_test


class Trainer(metaclass=ABCMeta):
    def __init__(self, model_class, energy_components, feature_group, energy_source, node_level, pipeline_name, scaler_type="maxabs"):
        self.energy_components = energy_components
        self.feature_group_name = feature_group
        self.feature_group = FeatureGroup[feature_group]
        self.features = FeatureGroups[self.feature_group]
        self.energy_source = energy_source
        self.node_level = node_level
        self.trainer_name = self.__class__.__name__
        self.model_class = model_class
        self.output_type = ModelOutputType.AbsPower if node_level else ModelOutputType.DynPower
        self.group_path = get_model_group_path(model_toppath, self.output_type, self.feature_group, self.energy_source, pipeline_name=pipeline_name)
        self.checkpoint_toppath = get_assured_checkpoint_path(self.group_path)
        self.node_models = dict()
        self.node_scalers = dict()
        self.scaler_type = scaler_type

    def _get_save_path(self, node_type):
        save_path = get_save_path(self.group_path, self.trainer_name, node_type=node_type)
        return assure_path(save_path)

    def _model_filename(self, node_type):
        model_name = get_model_name(self.trainer_name, node_type)
        model_file = model_name + ".json"
        return model_name, model_file

    def _checkpoint_filename(self, component, node_type):
        return "{}_{}_{}".format(self.trainer_name, component, node_type)

    def _checkpoint_filepath(self, component, node_type):
        checkpoint_filename = self._checkpoint_filename(component, node_type)
        return os.path.join(self.checkpoint_toppath, checkpoint_filename)

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
    def get_mape(self, node_type, component, X_test, y_test):
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
            if ok:
                self.node_models[node_type][component] = model
                self.print_log("Continue from last checkpoint ({})".format(component))
            else:
                # init if failed to load any checkpoint
                self.node_models[node_type][component] = self.init_model()
                self.print_log("Newly initialize model ({})".format(component))
            if hasattr(self.node_models[node_type][component], "set_feature_index"):
                feature_index = main_feature(self.feature_group_name, component)
                self.node_models[node_type][component].set_feature_index(feature_index)

    def process(self, data, power_labels, pipeline_lock):
        node_types = pd.unique(data[node_info_column])
        for node_type in node_types:
            node_type = int(node_type)
            save_path = self._get_save_path(str(node_type))
            self.node_scalers[node_type] = load_scaler(save_path)
            self.load_model(node_type)

            node_type_filtered_data = data[data[node_info_column] == node_type]
            if self.node_scalers[node_type] is None:
                self.print_log("fit scaler to latest data {1} for node_type={0}".format(node_type, self.feature_group_name))
                # no profiled scaler
                x_values = node_type_filtered_data[self.features].values
                self.node_scalers[node_type] = MaxAbsScaler()
                self.node_scalers[node_type].fit(x_values)

            X_test_map = dict()
            y_test_map = dict()
            try:
                for component in self.energy_components:
                    X_values, y_values = self.apply_ratio(component, node_type_filtered_data, power_labels)
                    X_train, X_test, y_train, y_test = normalize_and_split(X_values, y_values, scaler=self.node_scalers[node_type])
                    X_test_map[component] = X_test
                    y_test_map[component] = y_test
                    self.train(node_type, component, X_train, y_train)
                    self.save_checkpoint(self.node_models[node_type][component], self._checkpoint_filepath(component, node_type))
            except Exception as err:
                self.print_log("failed to process {}: {}".format(node_type, err))
                continue
            if self.should_archive(node_type):
                pipeline_lock.acquire()
                try:
                    self.save_model_and_metadata(node_type, X_test_map, y_test_map)
                except Exception as err:
                    self.print_log("failed to save model {}: {}".format(node_type, err))
                finally:
                    pipeline_lock.release()

    def apply_ratio(self, component, node_type_filtered_data, power_labels):
        power_label = component_to_col(component)
        related_labels = [label for label in power_labels if power_label in label]
        unit_vals = get_unit_vals(power_labels)
        if len(unit_vals) == 0:
            X_values = node_type_filtered_data[self.features].values
            y_values = node_type_filtered_data[related_labels].sum(axis=1)
            return X_values, y_values
        X_values = None
        y_values = None
        for unit_val in unit_vals:
            ratio_colname = ratio_to_col(unit_val)
            y_col = component_to_col(component, unit_col="package", unit_val=unit_val)
            if y_col not in node_type_filtered_data:
                # not define ratio
                y_col = component_to_col(component)
            multiplied_data = node_type_filtered_data[self.features].astype(float).copy()
            for feature in self.features:
                multiplied_data[feature] = node_type_filtered_data[feature] * node_type_filtered_data[ratio_colname]
            unit_X_values = multiplied_data.values
            unit_y_values = node_type_filtered_data[y_col].values
            if X_values is None:
                X_values = unit_X_values
            else:
                X_values += unit_X_values
            if y_values is None:
                y_values = unit_y_values
            else:
                y_values += unit_y_values
        return X_values, y_values

    def save_metadata(self, node_type, mae, mae_map, mape, mape_map, item):
        save_path = self._get_save_path(node_type)
        model_name, model_file = self._model_filename(node_type)
        item["model_name"] = model_name
        item["model_class"] = self.model_class
        item["model_file"] = model_file
        item["features"] = self.features
        item["fe_files"] = [] if not hasattr(self, "fe_files") else self.fe_files
        item["output_type"] = self.output_type.name
        item["mae"] = mae
        item["mape"] = mape
        item.update(mae_map)
        item.update(mape_map)
        self.metadata = item
        save_metadata(save_path, item)

    def archive_model(self, node_type):
        save_path = self._get_save_path(node_type)
        model_name, _ = self._model_filename(node_type)
        archived_file = get_archived_file(self.group_path, model_name)
        self.print_log("archive model :" + archived_file)
        self.print_log("save_path :" + save_path)
        shutil.make_archive(save_path, "zip", save_path)
        weight_dict = self.get_weight_dict(node_type)
        if weight_dict is not None:
            save_weight(save_path, weight_dict)

    def save_scaler(self, save_path, node_type):
        return save_scaler(save_path, self.node_scalers[node_type])

    def save_model_and_metadata(self, node_type, X_test_map, y_test_map):
        save_path = self._get_save_path(node_type)
        scaler_filename = self.save_scaler(save_path, node_type)

        _, model_dict_filename = self._model_filename(node_type)
        model_dict = dict()
        for component in self.energy_components:
            component_save_file = self.component_model_filename(component)
            model_dict[component] = {"model_file": component_save_file, "features": self.features, "fe_files": [scaler_filename] + ([] if not hasattr(self, "fe_files") else self.fe_files)}
            # save component model
            self.save_model(save_path, node_type, component)
        # save model dict
        save_json(save_path, model_dict_filename, model_dict)
        # save metadata
        max_mae = None
        max_mape = None
        mae_map = dict()
        mape_map = dict()
        item = self.get_basic_metadata(node_type)
        for component in self.energy_components:
            mae = self.get_mae(node_type, component, X_test_map[component], y_test_map[component])
            mae = round(mae, 2)
            mape = self.get_mape(node_type, component, X_test_map[component], y_test_map[component])
            mape = round(mape, 2)
            if max_mae is None or mae > max_mae:
                max_mae = mae
            if max_mape is None or mape > max_mape:
                max_mape = mape
            mae_map["{}_mae".format(component)] = mae
            mape_map["{}_mape".format(component)] = mape
        self.save_metadata(node_type, max_mae, mae_map, mape, mape_map, item)
        # archive model
        self.archive_model(node_type)
        print("save model to {}".format(save_path))

    def predict(self, node_type, component, X_values, skip_preprocess=False):
        save_path = self._get_save_path(node_type)
        if not skip_preprocess:
            node_scaler = load_scaler(save_path)
            if node_scaler is None:
                self.print_log("cannot predict because of no scaler/model")
                return None
            features = node_scaler.transform(X_values)
        else:
            features = X_values

        if hasattr(self, "fe"):
            for fe in self.fe:
                features = fe.transform(features)

        model = self.node_models[node_type][component]
        return model.predict(features)

    def print_log(self, message):
        print("{}: {}".format(self.to_string(), message), flush=True)

    def to_string(self):
        return "{} trainer ({}/{}/{})".format(self.trainer_name, "Abs" if self.node_level else "Dyn", self.feature_group, self.energy_source)

    def get_metadata(self):
        items = []
        for node_type in self.node_models.keys():
            save_path = self._get_save_path(node_type)
            item = load_metadata(save_path)
            item["node_type"] = node_type
            items += [item]
        return pd.DataFrame(items)

