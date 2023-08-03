
from config import getConfig
import pandas as pd
from train_types import SYSTEM_FEATURES, FeatureGroups, FeatureGroup, get_valid_feature_groups
PROM_SERVER = 'http://localhost:9090'
PROM_SSL_DISABLE = 'True'
PROM_HEADERS = ''
PROM_QUERY_INTERVAL = 300
PROM_QUERY_STEP = 3

PROM_SERVER = getConfig('PROM_SERVER', PROM_SERVER)
PROM_HEADERS = getConfig('PROM_HEADERS', PROM_HEADERS)
PROM_HEADERS = None if PROM_HEADERS == '' else PROM_HEADERS
PROM_SSL_DISABLE = True if getConfig('PROM_SSL_DISABLE', PROM_SSL_DISABLE).lower() == 'true' else False
PROM_QUERY_INTERVAL = getConfig('PROM_QUERY_INTERVAL', PROM_QUERY_INTERVAL)
PROM_QUERY_STEP = getConfig('PROM_QUERY_STEP', PROM_QUERY_STEP)

metric_prefix = "kepler_"
TIMESTAMP_COL = "timestamp"
PACKAGE_COL = "package"
SOURCE_COL = "source"
MODE_COL = "mode"

container_query_prefix = "kepler_container"
container_query_suffix = "total"

node_query_prefix = "kepler_node"
node_query_suffix = "joules_total"

usage_ratio_query = "kepler_container_cpu_usage_per_package_ratio"
# mostly available
valid_container_query = "kepler_container_kubelet_cpu_usage_total"
node_info_query = "kepler_node_node_info"
cpu_frequency_info_query = "kepler_node_cpu_scaling_frequency_hertz"

container_id_cols = ["pod_name", "container_name", "container_namespace"]
node_info_column = "node_type"
pkg_id_column = "pkg_id"

def get_energy_unit(component):
    if component in ["package", "core", "uncore", "dram"]:
        return "package"
    return None

def feature_to_query(feature):
    if feature in SYSTEM_FEATURES:
        return "{}_{}".format(node_query_prefix, feature)
    return "{}_{}_{}".format(container_query_prefix, feature, container_query_suffix)

def energy_component_to_query(component):
    return "{}_{}_{}".format(node_query_prefix, component, node_query_suffix)

def get_valid_feature_group_from_queries(queries):
    all_workload_features = FeatureGroups[FeatureGroup.WorkloadOnly]
    features = [feature for feature in all_workload_features if feature_to_query(feature) in queries]
    return get_valid_feature_groups(features)

def split_container_id_column(container_id):
    split_values = dict()
    splits = container_id.split('/')
    if len(splits) != len(container_id_cols):
        # failed to split
        return None
    index = 0 
    for col_name in container_id_cols:
        split_values[col_name] = splits[index]
        index += 1
    return split_values

def get_container_name_from_id(container_id):
    split_values = split_container_id_column(container_id)
    if split_values is None:
        return None
    return split_values["container_name"]

def generate_dataframe_from_response(query_metric, prom_response):
    items = []
    for res in prom_response:
        metric_item = res['metric']
        for val in res['values']:
            # labels
            item = metric_item.copy()
            # timestamp
            item[TIMESTAMP_COL] = val[0]
            # value
            item[query_metric] = float(val[1]) 
            items += [item]
    df = pd.DataFrame(items)
    return df

def prom_responses_to_results(prom_responses):
    results = dict()
    for query_metric, prom_response in prom_responses.items():
        results[query_metric] = generate_dataframe_from_response(query_metric, prom_response)
    return results