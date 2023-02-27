# extractor_test.py
#   call 

import os
import sys
from copy import deepcopy

server_path = os.path.join(os.path.dirname(__file__), '../server')
util_path = os.path.join(os.path.dirname(__file__), '../server/util')
train_path = os.path.join(os.path.dirname(__file__), '../server/train')
prom_path = os.path.join(os.path.dirname(__file__), '../server/prom')


sys.path.append(server_path)
sys.path.append(util_path)
sys.path.append(train_path)
sys.path.append(prom_path)


from train.extractor import DefaultExtractor
from train.train_types import FeatureGroups, FeatureGroup
from prom.query import PrometheusClient


from prom_test import prom_output_path


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

def extract_kubelet_features_package_power_label_test(query_results, extractor_instance):
    extracted_data = extractor_instance.extract(query_results, ["package", "core", "uncore", "dram"], 'KubeletOnly', "rapl", node_level=True)
    expected_col_size = len(["package", "core", "uncore", "dram"]) * 4 + len(FeatureGroups[FeatureGroup["KubeletOnly"]])
    assert len(extracted_data.columns) == expected_col_size, "Unmatched columns: expected {}, got {}".format(expected_col_size, len(extracted_data.columns))

def extract_bpf_features_package_power_label_node_test(query_results, extractor_instance):
    extracted_data_node_level = extractor_instance.extract(query_results, ["package", "dram"], 'IRQOnly', "rapl", node_level=True)
    assert(extracted_data_node_level is not None)
    print(extracted_data_node_level.columns.tolist())


def extract_bpf_features_package_power_label_container_test(query_results, extractor_instance):
    extracted_data_container_level = extractor_instance.extract(query_results, ["package", "dram"], 'IRQOnly', "rapl", node_level=False)
    assert(extracted_data_container_level is not None)
    print(extracted_data_container_level.columns.tolist())
    print(len(extracted_data_container_level))


if __name__ == '__main__':
    # Note that extractor mutates the query results
    query_results = read_sample_query_results()
    query_results_container = deepcopy(query_results)
    for extractor in test_extractors:
        extractor_instance = extractor()
        extract_bpf_features_package_power_label_node_test(query_results, extractor_instance)
        extract_bpf_features_package_power_label_container_test(query_results_container, extractor_instance)

        extract_kubelet_features_package_power_label_test(query_results, extractor_instance)
