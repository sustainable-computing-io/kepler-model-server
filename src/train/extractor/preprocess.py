import os
import sys

import pandas as pd
import numpy as np

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'estimate', 'model')
sys.path.append(model_path)

from train_types import FeatureGroup, FeatureGroups, PowerSourceMap
from prom_types import PROM_QUERY_STEP, TIMESTAMP_COL
from extract_types import container_id_colname, col_to_component
from model import get_label_power_colname

def drop_zero_column(data, cols):
    sum_col = "sum_val"
    data[sum_col] = data[cols].sum(axis=1)
    data = data.drop(data[data[sum_col] == 0].index)
    data = data.drop(columns=[sum_col])
    return data

def remove_outlier(df, workload_features, threshold=1):
    # Calculate the Z-score for each column
    z_scores = np.abs((df[workload_features] - df[workload_features].mean()) / df[workload_features].std())
    # Remove rows with outliers
    df_no_outliers = df[(z_scores < threshold).all(axis=1)]
    return df_no_outliers

def time_filter(data, min_time, max_time):
    _data = data.reset_index()
    start_time = _data[TIMESTAMP_COL].min()
    _data = _data[(_data[TIMESTAMP_COL] >= start_time+min_time) & (_data[TIMESTAMP_COL] <= start_time+max_time)]
    return _data

def correct_missing_metric_to_watt(feature_data, power_data, workload_features, power_columns):
    # remove missing metric

    # remove zero data
    # non_zero_feature_data = drop_zero_column(feature_data, workload_features)
    # feature_power_data = non_zero_feature_data.set_index(TIMESTAMP_COL).join(power_data).sort_index().dropna()
    feature_power_data = feature_data.set_index(TIMESTAMP_COL).join(power_data).sort_index().dropna()
    _feature_power_data = feature_power_data.groupby([TIMESTAMP_COL, container_id_colname]).sum().reset_index()
    # cgroup source data - aggr, divide over time difference
    # bpf source data - curr, divide over sampling period (PROM_QUERY_STEP)
    container_id_list = pd.unique(_feature_power_data[container_id_colname])
    container_df_list = []
    for container_id in container_id_list:
        container_df = _feature_power_data[_feature_power_data[container_id_colname]==container_id]
        container_df = container_df.sort_values(by=[TIMESTAMP_COL])
        time_diff = container_df[TIMESTAMP_COL].diff()
        for feature in workload_features:
            if feature in FeatureGroups[FeatureGroup.CounterIRQCombined]:
                container_df[feature] = container_df[feature]/PROM_QUERY_STEP
                # container_df[feature] = container_df[feature]/time_diff
            else:
                container_df[feature] = container_df[feature]/PROM_QUERY_STEP
        container_df[power_columns] = container_df[power_columns]/PROM_QUERY_STEP
        # for col in power_columns:
        #     container_df[col] = container_df[col]/time_diff
        container_df = container_df.dropna()

        container_df.set_index(TIMESTAMP_COL)
        container_df_list += [container_df]
    result_df = pd.concat(container_df_list).dropna()
    return result_df

def get_extracted_power_labels(extracted_data, energy_components, label_cols):
    # mean over the same value across container-level
    extracted_power_labels = extracted_data[[TIMESTAMP_COL] + label_cols].groupby([TIMESTAMP_COL]).mean().sort_index()
    for energy_component in energy_components:
        target_cols = [col for col in label_cols if col_to_component(col) == energy_component]
        component_label_col = get_label_power_colname(energy_component)
        extracted_power_labels[component_label_col] = extracted_power_labels[target_cols].sum(axis=1)
    return extracted_power_labels

def find_correlations(energy_source, feature_power_data, power_columns, workload_features):
    power_data = feature_power_data[power_columns].reset_index().groupby([TIMESTAMP_COL]).mean()
    feature_data = feature_power_data[workload_features].reset_index().groupby([TIMESTAMP_COL]).sum()
    energy_components = PowerSourceMap[energy_source]
    target_cols = [col for col in power_columns if col_to_component(col) == energy_components[0]]
    process_power_data = power_data.copy()
    # mean over the same value across container-level
    process_power_over_ts =  process_power_data[target_cols].reset_index().groupby([TIMESTAMP_COL]).sum()
    process_power_data[energy_source] = process_power_over_ts.sum(axis=1)
    # sum usage all container-level
    join_data = feature_data.join(process_power_data[energy_source]).dropna()
    corr = join_data.corr()[[energy_source]]
    return corr.drop(index=energy_source)