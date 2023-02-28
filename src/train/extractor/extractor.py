import os
import sys
import pandas as pd

prom_path = os.path.join(os.path.dirname(__file__), '../../../../prom')
sys.path.append(prom_path)

from abc import ABCMeta, abstractmethod
from prom.query import TIMESTAMP_COL, SOURCE_COL, get_energy_unit
from train_types import FeatureGroups, FeatureGroup, SYSTEM_FEATURES

container_query_prefix = "kepler_container"
container_query_suffix = "total"

node_query_prefix = "kepler_node"
node_query_suffix = "joules_total"

container_id_cols = ["container_name", "container_namespace"]
container_id_colname = "id"

node_info_query = "kepler_node_nodeInfo"
cpu_frequency_info_query = "kepler_node_cpu_scaling_frequency_hertz"
node_info_column = "node_type"
UNKNOWN_NODE_INFO = -1

def feature_to_query(feature):
    if feature in SYSTEM_FEATURES:
        return "{}_{}".format(node_query_prefix, feature)
    return "{}_{}_{}".format(container_query_prefix, feature, container_query_suffix)

def energy_component_to_query(component):
    return "{}_{}_{}".format(node_query_prefix, component, node_query_suffix)

def component_to_col(component, unit_col=None, unit_val=None):
    power_colname = "{}_power".format(component)
    if unit_col is None:
        return power_colname
    return "{}_{}_{}".format(unit_col, unit_val, power_colname)

def col_to_component(component_col):
    return component_col.split('_')[-2:][0]

class Extractor(metaclass=ABCMeta):
    # isolation abstract: should return dataFrame of features and labels
    @abstractmethod
    def extract(self, query_results, feature_group):
        return NotImplemented

# extract data from query 
# for node-level
# return DataFrame (index=timestamp, column=[features][power columns][node_type]), power_columns

class DefaultExtractor(Extractor):

    def get_workload_feature_data(self, query_results, features):
        feature_data_list = []
        for feature in features:
            query = feature_to_query(feature)
            if query not in query_results:
                print(query, "not in", list(query_results.keys()))
                return None
            aggr_query_data = query_results[query].copy()
            aggr_query_data.rename(columns={query: feature}, inplace=True)
            aggr_query_data[container_id_colname] = aggr_query_data[container_id_cols].apply(lambda x: '/'.join(x), axis=1)
            aggr_query_data.set_index([TIMESTAMP_COL, container_id_colname], inplace=True)
            # find current value from aggregated query
            df = aggr_query_data.sort_index()[feature].diff().dropna()
            df = df.mask(df.lt(0)).ffill().fillna(0).convert_dtypes()
            feature_data_list += [df]

        feature_data = pd.concat(feature_data_list, axis=1)
        # fill empty timestamp
        feature_data.fillna(0, inplace=True)
        # return with reset index for later aggregation
        return feature_data.reset_index()

    def get_system_feature_data(self, query_results, features):
        feature_data_list = []
        for feature in features:
            query = feature_to_query(feature)
            if query not in query_results:
                print(query, "not in", list(query_results.keys()))
                return None
            aggr_query_data = query_results[query].copy()
            aggr_query_data.rename(columns={query: feature}, inplace=True)
            aggr_query_data = aggr_query_data.groupby([TIMESTAMP_COL]).mean().sort_index()
            feature_data_list += [aggr_query_data]
        feature_data = pd.concat(feature_data_list, axis=1).astype(int)
        return feature_data

    # return with timestamp index
    def get_power_data(self, query_results, energy_components, source):
        power_data_list = []
        for component in energy_components:
            unit_col = get_energy_unit(component)
            query = energy_component_to_query(component)
            if query not in query_results:
                return None
            aggr_query_data = query_results[query].copy()
            # filter source
            aggr_query_data = aggr_query_data[aggr_query_data[SOURCE_COL] == source]
            if unit_col is not None:
                # sum over mode
                aggr_query_data = aggr_query_data.groupby([unit_col, TIMESTAMP_COL]).sum().reset_index().set_index(TIMESTAMP_COL)
                # add per unit_col
                unit_vals = pd.unique(aggr_query_data[unit_col])
                for unit_val in unit_vals:
                    df = aggr_query_data[aggr_query_data[unit_col]==unit_val].copy()
                    # rename
                    colname = component_to_col(component, unit_col, unit_val)
                    df.rename(columns={query: colname}, inplace=True)
                    # find current value from aggregated query
                    df = df.sort_index()[colname].diff().dropna()
                    df = df.mask(df.lt(0)).ffill().fillna(0).convert_dtypes()
                    power_data_list += [df]
            else:
                # sum over mode
                aggr_query_data = aggr_query_data.groupby([TIMESTAMP_COL]).sum()
                # rename
                colname = component_to_col(component)
                aggr_query_data.rename(columns={query: colname}, inplace=True)
                # find current value from aggregated query
                df = aggr_query_data.sort_index()[colname].diff().dropna()
                df = df.mask(df.lt(0)).ffill().fillna(0).convert_dtypes()
                power_data_list += [df]
        power_data = pd.concat(power_data_list, axis=1)
        # fill empty timestamp
        power_data.fillna(0, inplace=True)
        return power_data

    def get_system_category(self, query_results):
        node_info_data = None
        if node_info_query in query_results:
            node_info_data = query_results[node_info_query][[TIMESTAMP_COL, node_info_query]].set_index(TIMESTAMP_COL)
            node_info_data.rename(columns={node_info_query: node_info_column}, inplace=True)
        return node_info_data

    def extract(self, query_results, energy_components, feature_group, energy_source, node_level, aggr=True):
        power_data = self.get_power_data(query_results, energy_components, energy_source)
        if power_data is None:
            return None
        power_columns = power_data.columns
        features = FeatureGroups[FeatureGroup[feature_group]]
        workload_features = [feature for feature in features if feature not in SYSTEM_FEATURES]
        system_features = [feature for feature in features if feature in SYSTEM_FEATURES]
        feature_data = self.get_workload_feature_data(query_results, workload_features)
        if feature_data is None:
            return None
        if node_level and aggr:
            # sum stat of all containers
            workload_feature_data = feature_data.groupby([TIMESTAMP_COL]).sum()[workload_features]
        else:
            workload_feature_data = feature_data.groupby([TIMESTAMP_COL, container_id_colname]).sum()[workload_features]
        if len(system_features) > 0:
            system_feature_data = self.get_system_feature_data(query_results, system_features)
            feature_data = workload_feature_data.join(system_feature_data).sort_index().dropna()
        else: 
            feature_data = workload_feature_data
        feature_power_data = feature_data.join(power_data).sort_index().dropna()
        node_info_data = self.get_system_category(query_results)
        if node_info_data is None:
            feature_power_data[node_info_column] = UNKNOWN_NODE_INFO
        else:
            feature_power_data = feature_power_data.join(node_info_data)
        return  feature_power_data, power_columns

