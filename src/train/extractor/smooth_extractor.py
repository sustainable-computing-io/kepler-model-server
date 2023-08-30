import os 
import sys

from extractor import DefaultExtractor, find_correlations
util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

from train_types import FeatureGroups, FeatureGroup, SYSTEM_FEATURES

class SmoothExtractor(DefaultExtractor):

    def __init__(self, smooth_window=30):
        self.smooth_window = smooth_window

    # implement extract function
    def extract(self, query_results, energy_components, feature_group, energy_source, node_level, aggr=True):
        feature_power_data, power_columns, _, features = super().extract(query_results, energy_components, feature_group, energy_source, node_level, aggr)

        features = FeatureGroups[FeatureGroup[feature_group]]
        smoothed_data = feature_power_data.copy()
        workload_features = [feature for feature in features if feature not in SYSTEM_FEATURES]

        for col in list(workload_features) + list(power_columns):
            smoothed_data[col] = feature_power_data[col].rolling(window=self.smooth_window).mean()
        smoothed_data = smoothed_data.dropna()

        corr = find_correlations(energy_source, feature_power_data, power_columns, workload_features)

        return smoothed_data, power_columns, corr, features

