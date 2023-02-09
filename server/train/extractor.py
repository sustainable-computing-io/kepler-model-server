import os
import sys
import pandas as pd

prom_path = os.path.join(os.path.dirname(__file__), '../../../prom')
sys.path.append(prom_path)

from abc import ABCMeta, abstractmethod
from prom.query import TIMESTAMP_COL, SOURCE_COL, get_energy_unit
from train_types import FeatureGroups, FeatureGroup

container_query_prefix = "kepler_container"
container_query_suffix = "total"

node_query_prefix = "kepler_node"
node_query_suffix = "joules_total"

container_id_cols = ["container_name", "container_namespace"]
container_id_colname = "id"

def feature_to_query(feature):
    return "{}_{}_{}".format(container_query_prefix, feature, container_query_suffix)

def energy_component_to_query(component):
    return "{}_{}_{}".format(node_query_prefix, component, node_query_suffix)

def component_to_col(component, unit_col=None, unit_val=None):
    power_colname = "{}_power".format(component)
    if unit_col is None:
        return power_colname
    return "{}_{}_{}".format(unit_col, unit_val, power_colname)


class Extractor(metaclass=ABCMeta):
    # isolation abstract: should return dataFrame of features and labels
    @abstractmethod
    def extract(self, query_results, feature_group):
        return NotImplemented

# extract data from query 
class DefaultExtractor(Extractor):

    def get_feature_data(self, query_results, features):
        feature_data_list = []
        for feature in features:
            query = feature_to_query(feature)
            if query not in query_results:
                return None
            aggr_query_data = query_results[query]
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

    # return with timestamp index
    def get_power_data(self, query_results, energy_components, source):
        power_data_list = []
        for component in energy_components:
            unit_col = get_energy_unit(component)
            query = energy_component_to_query(component)
            if query not in query_results:
                return None
            aggr_query_data = query_results[query]
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

    def extract(self, query_results, energy_components, feature_group, energy_source, node_level=False):
        power_data = self.get_power_data(query_results, energy_components, energy_source)
        if power_data is None:
            return None
        features = FeatureGroups[FeatureGroup[feature_group]]
        feature_data = self.get_feature_data(query_results, features)
        if feature_data is None:
            return None
        if node_level:
            # combined all containers
            feature_data = feature_data.groupby([TIMESTAMP_COL]).sum()[features]
        return  pd.concat([feature_data, power_data], axis=1).sort_index().dropna()

