############################################################
##
## profile_background
## generate a profile from prom query
## 
## ./python profile_background.py query_output_folder
## e.g., ./python profile_background.py ../tests/prom_output
##
## input must be a query output of idle state
##
## profile saved in data/profile
##  [source].json
##   {component: {node_type: {min_watt: ,max_watt: } }}
############################################################

import sys
import os

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

from train_types import PowerSourceMap, FeatureGroups
from train_types import  PowerSourceMap
from prom_types import node_info_column, node_info_query
from extract_types import component_to_col

import pandas as pd
import json

min_watt_key = "min_watt"
max_watt_key = "max_watt"

resource_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'resource')
profile_top_path = os.path.join(resource_path, 'profiles')
profile_path = os.path.join(profile_top_path, "profile")

profiler_registry = "https://raw.githubusercontent.com/sustainable-computing-io/kepler-model-db/main/profiles"

if not os.path.exists(resource_path):
    os.mkdir(resource_path)

if not os.path.exists(profile_top_path):
    os.mkdir(profile_top_path)

if not os.path.exists(profile_path):
    os.mkdir(profile_path)

def read_query_results(query_path):
    results = dict()
    metric_filenames = [ metric_filename for metric_filename in os.listdir(query_path) ]
    for metric_filename in metric_filenames:
        metric = metric_filename.replace(".csv", "")
        filepath = os.path.join(query_path, metric_filename)
        results[metric] = pd.read_csv(filepath)
    return results

def load_profile(source):
    profile_filename = os.path.join(profile_path, source + ".json")
    if not os.path.exists(profile_filename):
        profile = dict()
        for component in PowerSourceMap[source]:
            profile[component] = dict()
    else:
        with open(profile_filename) as f:
            profile = json.load(f)
    return profile

def save_profile(profile, source):
    profile_filename = os.path.join(profile_path, source + ".json")
    with open(profile_filename, "w") as f:
        json.dump(profile, f)

def get_min_max_watt(profiles, component, node_type):
    profile = profiles[component][node_type]
    return profile[min_watt_key], profile[max_watt_key]

def response_to_result(response):
    results = dict()
    for query in response.keys():
        results[query] = generate_dataframe_from_response(query, response[query])
        if len(results[query]) > 0:
            if query == node_info_query:
                results[query][query] = results[query][query].astype(int)
            else:
                results[query][query] = results[query][query].astype(float)
    return results

class Profiler():
    def __init__(self, extractor):
        self.extractor = extractor

    def process(self, query_results, save=True):
        node_types, node_info_data = self.extractor.get_node_types(query_results)
        if node_info_data is None:
            return None
        result = dict()
        for source, energy_components in PowerSourceMap.items():
            # profile = load_profile(source)
            profile = dict()
            for node_type in node_types:
                power_data= self.extractor.get_power_data(query_results, energy_components, source)
                power_data = power_data.join(node_info_data)
                power_labels = power_data.columns
                for component in energy_components:
                    power_label = component_to_col(component) 
                    related_labels = [label for label in power_labels if power_label in label]
                    # filter and extract features
                    power_values = power_data[power_data[node_info_column]==node_type][related_labels].min(axis=1) # minimum single unit powerc
                    time_values = power_data.index.values
                    seconds = time_values[1] - time_values[0]
                    max_watt = power_values.max()/seconds
                    min_watt = power_values.min()/seconds
                    node_type_key = str(int(node_type))
                    print(component, node_type, min_watt, seconds)
                    if component not in profile:
                        profile[component] = dict()
                    if node_type_key not in profile[component]:
                        profile[component][node_type_key] = {
                            min_watt_key: min_watt,
                            max_watt_key: max_watt
                        }
                    else:
                        if min_watt < profile[component][node_type_key][min_watt_key]:
                            profile[component][node_type_key][min_watt_key] = min_watt
                        if max_watt > profile[component][node_type_key][max_watt_key]:
                            profile[component][node_type_key][max_watt_key] = max_watt
                    print("update:", component, node_type_key, min_watt_key, profile[component][node_type_key][min_watt_key])
            print(profile)
            if save:
                save_profile(profile, source)
            result[source] = profile 
        return result


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



