import os
import sys

import numpy as np

util_path = os.path.join(os.path.dirname(__file__), "..", "..", "util")
sys.path.append(util_path)

model_path = os.path.join(os.path.dirname(__file__), "..", "..", "estimate", "model")
sys.path.append(model_path)

from train_types import PowerSourceMap
from prom_types import TIMESTAMP_COL
from extract_types import col_to_component
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


def get_extracted_power_labels(extracted_data, energy_components, label_cols):
    # mean over the same value across container-level
    extracted_power_labels = extracted_data[[TIMESTAMP_COL] + label_cols].groupby([TIMESTAMP_COL]).mean().sort_index()
    for energy_component in energy_components:
        target_cols = [col for col in label_cols if col_to_component(col) == energy_component]
        component_label_col = get_label_power_colname(energy_component)
        extracted_power_labels[component_label_col] = extracted_power_labels[target_cols].sum(axis=1)
    return extracted_power_labels


def find_correlations(energy_source, feature_power_data, power_columns, workload_features):
    power_data = feature_power_data.reset_index().groupby([TIMESTAMP_COL])[power_columns].mean()
    feature_data = feature_power_data.reset_index().groupby([TIMESTAMP_COL])[workload_features].sum()
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
