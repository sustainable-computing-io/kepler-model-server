import os
import sys
cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)

util_path = os.path.join(os.path.dirname(__file__), '..', 'util')
sys.path.append(util_path)
extractor_path = os.path.join(os.path.dirname(__file__), 'extractor')
sys.path.append(extractor_path)
isolator_path = os.path.join(os.path.dirname(__file__), 'isolator')
sys.path.append(isolator_path)

from extractor import DefaultExtractor
from isolator import MinIdleIsolator, ProfileBackgroundIsolator, NoneIsolator
from train_isolator import TrainIsoltor
from pipeline import NewPipeline

from urllib.request import urlopen
import json
import joblib

from pipeline import load_class

from util import PowerSourceMap, FeatureGroups

profiler_registry = "https://raw.githubusercontent.com/sustainable-computing-io/kepler-model-db/main/profiles"

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
            # standard_scaler = Profile.load_scaler(self.node_type, feature_key, scaler_type="standard")
            # minmax_scaler = Profile.load_scaler(self.node_type, feature_key, scaler_type="minmax")
            standard_scaler = None
            minmax_scaler = None
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

    def get_background_power(self, source, component):
        if source not in self.profile:
            return None
        if component not in self.profile[source]:
            return None
        background_power = (self.profile[source][component]["min_watt"] + self.profile[source][component]["max_watt"])/2
        return background_power
    
    def get_min_power(self, source, component):
        return self.profile[source][component]["min_watt"]

    def print_profile(self):
        print("Profile (node type={}): \n Available energy components: {}\n Available minmax scalers: {}\n Available standard scalers: {}".format(self.node_type, ["{}/{}".format(key, list(self.profile[key].keys())) for key in self.profile.keys()], self.minmax_scaler.keys(), self.standard_scaler.keys()))

def load_all_profiles():
    profile_map = dict()
    for source in PowerSourceMap.keys():
        try:
            url_path = os.path.join(profiler_registry, "profile", source + ".json")
            response = urlopen(url_path)
            profile = json.loads(response.read())
        except Exception as e:
            print("Failed to load profile {}: {}".format(source, e))
            continue
        profile_map[source] = profile
    return generate_profiles(profile_map)

def generate_profiles(profile_map):
    profiles = dict()
    for source, profile in profile_map.items():
        for component, values in profile.items():
            for node_type, profile_value in values.items():
                node_type = int(float(node_type))
                if node_type not in profiles:
                    profiles[node_type] = Profile(node_type)
                profiles[node_type].add_profile(source, component, profile_value)
    for profile in profiles.values():
        profile.print_profile()
    return profiles