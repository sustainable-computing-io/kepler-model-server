# extractor_test.py
#   call 

import os
import sys

server_path = os.path.join(os.path.dirname(__file__), '../server')
util_path = os.path.join(os.path.dirname(__file__), '../server/util')
train_path = os.path.join(os.path.dirname(__file__), '../server/train')

sys.path.append(server_path)
sys.path.append(util_path)
sys.path.append(train_path)

from train.extractor import DefaultExtractor
from train.train_types import FeatureGroups, FeatureGroup

from prom_test import prom_output_path

energy_components = ["package", "core", "uncore", "dram"]
num_of_package = 4
feature_group = 'KubeletOnly'
energy_source = "rapl"

test_extractors = [DefaultExtractor]

import pandas as pd

def read_sample_query_results():
    results = dict()
    metric_filenames = [ metric_filename for metric_filename in os.listdir(prom_output_path) ]
    for metric_filename in metric_filenames:
        metric = metric_filename.replace(".csv", "")
        filepath = os.path.join(prom_output_path, metric_filename)
        results[metric] = pd.read_csv(filepath)
    return results

if __name__ == '__main__':
    query_results = read_sample_query_results()
    for extractor in test_extractors:
        extractor_instance = extractor()
        extracted_data = extractor_instance.extract(query_results, energy_components, feature_group, energy_source, node_level=True)
        expected_col_size = len(energy_components) * num_of_package + len(FeatureGroups[FeatureGroup[feature_group]])
        print(extracted_data)
        assert len(extracted_data.columns) == expected_col_size, "Unmatched columns: expected {}, got {}".format(expected_col_size, len(extracted_data.columns))
        