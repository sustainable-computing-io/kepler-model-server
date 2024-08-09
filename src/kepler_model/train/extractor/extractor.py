import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod

from kepler_model.util.prom_types import TIMESTAMP_COL, SOURCE_COL, get_energy_unit, usage_ratio_query, node_info_query, energy_component_to_query, feature_to_query, pkg_id_column, container_id_cols, node_info_column
from kepler_model.util.train_types import FeatureGroups, FeatureGroup, SYSTEM_FEATURES
from kepler_model.util.loader import default_node_type
from kepler_model.util.extract_types import container_id_colname, ratio_to_col, component_to_col, get_unit_vals, accelerator_type_colname
from kepler_model.train.extractor.preprocess import drop_zero_column, find_correlations


# append ratio for each unit
def append_ratio_for_pkg(feature_power_data, is_aggr, query_results, power_columns):
    unit_vals = get_unit_vals(power_columns)
    if len(unit_vals) == 0:
        # not relate/not append
        return feature_power_data
    use_default_ratio = False
    default_ratio = 1 / len(unit_vals)
    if usage_ratio_query not in query_results:
        use_default_ratio = True
    else:
        ratio_df = query_results[usage_ratio_query]
        if is_aggr:
            ratio_df = ratio_df.groupby([TIMESTAMP_COL, pkg_id_column]).sum()[usage_ratio_query]
        else:
            ratio_df[container_id_colname] = ratio_df[container_id_cols].apply(lambda x: "/".join(x), axis=1)
            ratio_df = ratio_df.groupby([TIMESTAMP_COL, pkg_id_column, container_id_colname]).sum()[usage_ratio_query]
    ratio_colnames = []
    for unit_val in unit_vals:
        ratio_colname = ratio_to_col(unit_val)
        if use_default_ratio:
            feature_power_data[ratio_colname] = default_ratio
        else:
            target_ratio_df = ratio_df.xs(unit_val, level=1)
            feature_power_data = feature_power_data.join(target_ratio_df).dropna()
            feature_power_data = feature_power_data.rename(columns={usage_ratio_query: ratio_colname})
        ratio_colnames += [ratio_colname]
    tmp_total_col = "total_ratio"
    feature_power_data[tmp_total_col] = feature_power_data[ratio_colnames].sum(axis=1)
    for ratio_colname in ratio_colnames:
        feature_power_data[ratio_colname] /= feature_power_data[tmp_total_col]
    return feature_power_data.drop(columns=[tmp_total_col])


class Extractor(metaclass=ABCMeta):
    # extractor abstract:
    # return
    # - feature_power_data: dataframe of feature columns concat with power columns
    # - power_columns: identify power columns (labels)
    # - corr: correlation matrix between features and powers
    # - features: updated features
    @abstractmethod
    def extract(self, query_results, feature_group):
        return NotImplemented

    # return short name
    @abstractmethod
    def get_name(self):
        return NotImplemented


# extract data from query
# for node-level
# return DataFrame (index=timestamp, column=[features][power columns][node_type]), power_columns


class DefaultExtractor(Extractor):
    def get_name(self):
        return "default"

    # implement extract function
    def extract(self, query_results, energy_components, feature_group, energy_source, node_level, aggr=True):
        # 1. compute energy different per timestamp and concat all energy component and unit
        power_data = self.get_power_data(query_results, energy_components, energy_source)
        if power_data is None:
            return None, None, None, None
        power_data = drop_zero_column(power_data, power_data.columns)
        power_columns = power_data.columns
        fg = FeatureGroup[feature_group]
        features = FeatureGroups[fg]
        # 2. separate workload and system
        workload_features = [feature for feature in features if feature not in SYSTEM_FEATURES]
        system_features = [feature for feature in features if feature in SYSTEM_FEATURES]
        # 3. compute aggregated utilization different per timestamp and concat them
        if fg == FeatureGroup.AcceleratorOnly and node_level is not True:
            return None, None, None, None
        else:
            feature_data, workload_features = self.get_workload_feature_data(query_results, workload_features)

        if feature_data is None:
            return None, None, None, None

        # join power
        feature_power_data = feature_data.set_index(TIMESTAMP_COL).join(power_data).sort_index().dropna()

        # aggregate data if needed
        is_aggr = node_level and aggr
        if is_aggr:
            # sum stat of all containers
            sum_feature = feature_power_data.groupby([TIMESTAMP_COL])[workload_features].sum()
            mean_power = feature_power_data.groupby([TIMESTAMP_COL])[power_columns].mean()
            feature_power_data = sum_feature.join(mean_power)
        else:
            feature_power_data = feature_power_data.groupby([TIMESTAMP_COL, container_id_colname]).sum()

        # 4. add system features (non aggregated data)
        if len(system_features) > 0:
            system_feature_data = self.get_system_feature_data(query_results, system_features)
            feature_power_data = feature_power_data.join(system_feature_data).sort_index().dropna()
        else:
            feature_power_data = feature_power_data.sort_index()

        # 5. add node info data
        node_info_data = self.get_system_category(query_results)
        if node_info_data is not None:
            feature_power_data = feature_power_data.join(node_info_data)
        if node_info_column not in feature_power_data.columns:
            feature_power_data[node_info_column] = default_node_type
        feature_power_data[node_info_column] = feature_power_data[node_info_column].astype(int)

        # 6. validate input with correlation
        corr = find_correlations(energy_source, feature_power_data, power_columns, workload_features)
        # 7. apply utilization ratio to each power unit because the power unit is summation of all container utilization
        feature_power_data = append_ratio_for_pkg(feature_power_data, is_aggr, query_results, power_columns)
        return feature_power_data, power_columns, corr, workload_features

    def get_workload_feature_data(self, query_results, features):
        feature_data = None
        container_df_map = dict()
        accelerator_df_list = []
        cur_accelerator_features = []
        feature_to_remove = []
        for feature in features:
            query = feature_to_query(feature)
            if query not in query_results:
                print(query, "not in", list(query_results.keys()))
                return None
            if len(query_results[query]) == 0:
                print("no data in ", query)
                return None
            aggr_query_data = query_results[query].copy()

            if all(col in aggr_query_data.columns for col in container_id_cols):
                aggr_query_data.rename(columns={query: feature}, inplace=True)
                aggr_query_data[container_id_colname] = aggr_query_data[container_id_cols].apply(lambda x: "/".join([str(xi) for xi in x]), axis=1)
                # separate for each container_id
                container_id_list = pd.unique(aggr_query_data[container_id_colname])

                for container_id in container_id_list:
                    container_df = aggr_query_data[aggr_query_data[container_id_colname] == container_id]
                    container_df = container_df.set_index([TIMESTAMP_COL])[[feature]]
                    container_df = container_df.sort_index()
                    time_diff_values = container_df.reset_index()[[TIMESTAMP_COL]].diff().values
                    feature_df = pd.DataFrame()
                    if len(container_df) > 1:
                        # find current value from aggregated query, dropna remove the first value
                        # divided by time difference
                        feature_df = container_df[[feature]].astype(np.float64).diff()
                        # if delta < 0, set to 0 (unexpected)
                        feature_df = feature_df / time_diff_values
                        feature_df = feature_df.mask(feature_df.lt(0)).ffill().fillna(0).convert_dtypes()
                        if container_id in container_df_map:
                            # previously found container
                            container_df_map[container_id] = pd.concat([container_df_map[container_id], feature_df], axis=1)
                        else:
                            # newly found container
                            container_df_map[container_id] = feature_df
            else:
                # process AcceleratorOnly fearure
                tmp_data_list = []
                feature_to_remove.append(feature)
                # separate based on type label
                grouped = aggr_query_data.groupby([accelerator_type_colname])
                for group_name, group_data in grouped:
                    new_colname = "{}_{}".format(feature, group_name)
                    cur_accelerator_features.append(new_colname)
                    group_data.rename(columns={query: new_colname}, inplace=True)
                    group_data = group_data[[TIMESTAMP_COL, new_colname]]
                    group_data = group_data.groupby([TIMESTAMP_COL]).sum().sort_index()
                    tmp_data_list += [group_data]
                accelerator_df_list += tmp_data_list

        container_df_list = []
        for container_id, container_df in container_df_map.items():
            container_df[container_id_colname] = container_id
            container_df_list += [container_df]

        sum_df_list = container_df_list + accelerator_df_list
        feature_data = pd.concat(sum_df_list)
        # fill empty timestamp
        feature_data.fillna(0, inplace=True)
        # update feature
        if len(feature_to_remove) != 0:
            features = self.process_feature(features, feature_to_remove, cur_accelerator_features)
        # return with reset index for later aggregation
        return feature_data.reset_index(), features

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
            unit_col = get_energy_unit(component)  # such as package
            query = energy_component_to_query(component)
            if query not in query_results:
                print(query, "not in", query_results)
                return None
            aggr_query_data = query_results[query].copy()
            # filter source
            aggr_query_data = aggr_query_data[aggr_query_data[SOURCE_COL] == source]
            if len(aggr_query_data) == 0:
                return None
            if unit_col is not None:
                if usage_ratio_query not in query_results:
                    # sum over mode (idle, dynamic) and unit col
                    df = aggr_query_data.groupby([TIMESTAMP_COL]).sum().reset_index().set_index(TIMESTAMP_COL)
                    time_diff_values = df.reset_index()[[TIMESTAMP_COL]].diff().dropna().values.mean()
                    df = df.loc[:, df.columns != unit_col]
                    # rename
                    colname = component_to_col(component)
                    df.rename(columns={query: colname}, inplace=True)
                    # find current value from aggregated query
                    df = df.sort_index()[colname].diff().dropna()
                    df /= time_diff_values
                    df = df.mask(df.lt(0)).ffill().fillna(0).convert_dtypes()
                    power_data_list += [df]
                else:
                    # sum over mode (idle, dynamic)
                    aggr_query_data = aggr_query_data.groupby([unit_col, TIMESTAMP_COL]).sum().reset_index().set_index(TIMESTAMP_COL)
                    time_diff_values = aggr_query_data.reset_index()[[TIMESTAMP_COL]].diff().dropna().values.mean()
                    # add per unit_col
                    unit_vals = pd.unique(aggr_query_data[unit_col])
                    for unit_val in unit_vals:
                        df = aggr_query_data[aggr_query_data[unit_col] == unit_val].copy()
                        # rename
                        colname = component_to_col(component, unit_col, unit_val)
                        df.rename(columns={query: colname}, inplace=True)
                        # find current value from aggregated query
                        df = df.sort_index()[colname].diff().dropna()
                        df /= time_diff_values
                        df = df.mask(df.lt(0)).ffill().fillna(0).convert_dtypes()
                        power_data_list += [df]
            else:
                # sum over mode
                aggr_query_data = aggr_query_data.groupby([TIMESTAMP_COL]).sum()
                time_diff_values = aggr_query_data.reset_index()[[TIMESTAMP_COL]].diff().dropna().values.mean()
                # rename
                colname = component_to_col(component)
                aggr_query_data.rename(columns={query: colname}, inplace=True)
                # find current value from aggregated query
                df = aggr_query_data.sort_index()[colname].diff().dropna()
                df /= time_diff_values
                df = df.mask(df.lt(0)).ffill().fillna(0).convert_dtypes()
                power_data_list += [df]
        if len(power_data_list) == 0:
            return None
        power_data = pd.concat(power_data_list, axis=1).dropna()
        return power_data

    def get_system_category(self, query_results):
        node_info_data = None
        if node_info_query in query_results:
            node_info_data = query_results[node_info_query][[TIMESTAMP_COL, node_info_query]].set_index(TIMESTAMP_COL)
            node_info_data.rename(columns={node_info_query: node_info_column}, inplace=True)
        return node_info_data

    def get_node_types(self, query_results):
        node_info_data = self.get_system_category(query_results)
        if node_info_data is None:
            print("No Node Info")
            return None, None
        return pd.unique(node_info_data[node_info_column]), node_info_data

    def process_feature(self, features, feature_to_remove, feature_to_add):
        new_features = []
        for feature in features:
            if feature not in feature_to_remove:
                new_features.append(feature)
        return new_features + feature_to_add
