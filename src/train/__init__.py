import sys
import os

from urllib.request import urlopen
import json
import joblib

util_path = os.path.join(os.path.dirname(__file__), '..', 'util')
sys.path.append(util_path)

from extractor.extractor import DefaultExtractor, node_info_column, component_to_col, TIMESTAMP_COL, UNKNOWN_NODE_INFO
from isolator.isolator import MinIdleIsolator, ProfileBackgroundIsolator
from train_types import ModelOutputType, PowerSourceMap, FeatureGroups, FeatureGroup, get_feature_group, is_weight_output, get_valid_feature_groups
from pipeline import Pipeline

profiler_registry = "https://raw.githubusercontent.com/sustainable-computing-io/kepler-model-db/main/profiles"

def load_class(module_name, class_name):
    path = os.path.join(os.path.dirname(__file__), '{}/{}'.format(module_name, class_name))
    sys.path.append(path)
    import importlib
    module_path = importlib.import_module('train.{}.{}.main'.format(module_name, class_name))
    return getattr(module_path, class_name)

class Profile:
    def __init__(self, node_type):
        self.node_type = node_type
        self.profile = dict()
        for source in PowerSourceMap.keys():
            self.profile[source] = dict()
        self.standard_scaler = dict()
        self.minmax_scaler = dict()
        for feature_group in FeatureGroups.keys():
            feature_key = feature_group.name
            standard_scaler = Profile.load_scaler(self.node_type, feature_key, scaler_type="standard")
            minmax_scaler = Profile.load_scaler(self.node_type, feature_key, scaler_type="minmax")
            if standard_scaler is not None:
                self.standard_scaler[feature_group.name] = standard_scaler
            if minmax_scaler is not None:
                self.minmax_scaler[feature_group.name] = minmax_scaler

    def add_profile(self, source, component, profile_value):
        self.profile[source][component] = profile_value
        
    @staticmethod
    def load_scaler(node_type, feature_key, scaler_type): # scaler_type = minmax or standard
        try:
            url_path = os.path.join(profiler_registry, scaler_type + "_scaler", str(node_type), feature_key + ".pkl")
            response = urlopen(url_path)
            scaler = joblib.load(response)
            return scaler
        except Exception as e:
            print(url_path, e)
            return None

    def get_minmax_scaler(self, feature_key):
        if feature_key not in self.minmax_scaler:
            return None
        return self.minmax_scaler[feature_key]

    def get_standard_scaler(self, feature_key):
        if feature_key not in self.standard_scaler:
            return None
        return self.standard_scaler[feature_key]

    def get_background_power(self, source, component, period=3):
        if source not in self.profile:
            return None
        if component not in self.profile[source]:
            return None
        background_power = self.profile[source][component]["min_watt"] * period
        return background_power

    def print_profile(self):
        print("Profile (node type={}): \n Available energy components: {}\n Available minmax scalers: {}\n Available standard scalers: {}".format(self.node_type, ["{}/{}".format(key, list(self.profile[key].keys())) for key in self.profile.keys()], self.minmax_scaler.keys(), self.standard_scaler.keys()))

def load_all_profiles():
    profiles = dict()
    for source in PowerSourceMap.keys():
        try:
            url_path = os.path.join(profiler_registry, "profile", source + ".json")
            response = urlopen(url_path)
            profile = json.loads(response.read())
        except Exception as e:
            print("Failed to load profile {}: {}".format(source, e))
            continue
        for component, values in profile.items():
            for node_type, profile_value in values.items():
                node_type = int(node_type)
                if node_type not in profiles:
                    profiles[node_type] = Profile(node_type)
                profiles[node_type].add_profile(source, component, profile_value)
    for profile in profiles.values():
        profile.print_profile()
    return profiles

def NewPipeline(pipeline_name, trainers, extractor=DefaultExtractor(), isolator=MinIdleIsolator()):
    return Pipeline(pipeline_name, trainers, extractor, isolator)
    
