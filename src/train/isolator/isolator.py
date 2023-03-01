import os
import sys
import pandas as pd

server_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(server_path)

from abc import ABCMeta, abstractmethod
from extractor.extractor import container_id_colname, TIMESTAMP_COL, col_to_component, node_info_column
from train_types import PowerSourceMap

container_indexes = [TIMESTAMP_COL, container_id_colname]

class Isolator(metaclass=ABCMeta):
    # isolation abstract: should return dataFrame of features and labels
    @abstractmethod
    def isolate(self, data, profile=None):
        return NotImplemented

def exclude_target_container_usage(data, target_container_id):
    target_container_data = data[data[container_id_colname]==target_container_id]
    filled_target_container_data = data[target_container_data.columns].join(target_container_data, lsuffix='_target').fillna(0)
    filled_target_container_data.drop(columns=[col for col in filled_target_container_data.columns if '_target' in col], inplace=True)
    conditional_data = data - filled_target_container_data
    return target_container_data, conditional_data

class MinIdleIsolator(Isolator):
    def isolate(self, data, energy_source, label_cols):
        isolated_data = data.copy()
        for label_col in label_cols:
            min = data[label_col].min()
            isolated_data[label_col] = data[label_col] - min
        return isolated_data

import numpy as np

system_process_id = "system_processes/system"

class ProfileBackgroundIsolator(Isolator):
    def __init__(self, profiles):
        self.profiles = profiles

    def transform_profile(self, node_type, energy_source, component):
        if node_type not in self.profiles:
            return np.nan    
        return self.profiles[node_type].get_background_power(energy_source, component)

    def transform_component(self, label_col):
        return col_to_component(label_col)

    def isolate(self, data, energy_source, label_cols):
        isolated_data = data.copy()
        isolated_data.iloc[isolated_data.index.get_level_values(container_id_colname) == system_process_id]
        try:
            for label_col in label_cols:
                component = col_to_component(label_col)
                isolated_data['profile'] = isolated_data[node_info_column].transform(self.transform_profile, energy_source=energy_source, component=component)
                if isolated_data['profile'].isnull().values.any():
                    return None
                isolated_data[label_col] = data[label_col] - isolated_data['profile']
                isolated_data.drop(columns='profile', inplace=True)
            return isolated_data
        except Exception as e:
            print(e)
            return None