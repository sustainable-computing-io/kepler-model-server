# plot.py
# to visualize data

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import sys

#################################################################
# import internal src 
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)
#################################################################

from util import assure_path, FeatureGroups, FeatureGroup, PowerSourceMap
from util.prom_types import TIMESTAMP_COL
from util.extract_types import col_to_component

from sklearn.preprocessing import MinMaxScaler

from train.extractor.preprocess import get_extracted_power_labels
from estimate import get_label_power_colname

plot_output_path = os.path.join(os.path.dirname(__file__), 'data', 'plot_output')
assure_path(plot_output_path)

def _fig_filename(figname, save_path=plot_output_path):
    return os.path.join(save_path, figname + ".png")

def preprocess_data(df):
    scaler = MinMaxScaler()
    df = df.reset_index()
    normalized_data = scaler.fit_transform(df.values)
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)
    return normalized_df

# plot extract result
from extractor_test import test_energy_source, get_extract_results, get_expected_power_columns, test_extractors
def plot_extract_result(extractor_name, feature_group, result, energy_source=test_energy_source, label_cols=get_expected_power_columns(), save_path=plot_output_path, features=None, title=None):
    energy_components = PowerSourceMap[energy_source]
    extracted_power_labels = get_extracted_power_labels(result, energy_components, label_cols)
    extracted_power_labels = preprocess_data(extracted_power_labels[5:])
    result = preprocess_data(result[5:])
    if features is None:
        features = FeatureGroups[FeatureGroup[feature_group]]
    ncols = max(len(features), len(energy_components))
    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(5+2*len(features),5))
    axes = np.array(axes).ravel()
    i = 0
    for feature in features:
        sns.lineplot(data=result, x=TIMESTAMP_COL, y=feature, ax=axes[i])
        axes[i].set_title("{}".format(feature))
        axes[i].set_ylabel("")
        i += 1
    while i < ncols:
        fig.delaxes(axes[i]) 
        i += 1
    i = ncols
    for energy_component in energy_components:
        component_label_col = get_label_power_colname(energy_component)
        sns.lineplot(data=extracted_power_labels, x=TIMESTAMP_COL, y=component_label_col, ax=axes[i])
        axes[i].set_title("{}".format(component_label_col))
        axes[i].set_ylabel("")
        i += 1
    while i < 2*ncols:
        fig.delaxes(axes[i]) 
        i += 1
    if title is None:
        title = "{} on {}".format(extractor_name, feature_group)
    plt.suptitle(title)
    

    figname = "extract_result_{}_{}".format(extractor_name, feature_group)
    plt.tight_layout()
    fig.savefig(_fig_filename(figname, save_path=save_path))

def plot_power_cols(extractor_name, result, energy_source=test_energy_source, label_cols=get_expected_power_columns(), save_path=plot_output_path):
    energy_components = PowerSourceMap[energy_source]
    fig = plt.figure()
    target_cols = [col for col in label_cols if col_to_component(col) in energy_components]
    target_df = result[[TIMESTAMP_COL] + target_cols].groupby([TIMESTAMP_COL]).mean().reset_index().sort_index()
    df = pd.melt(target_df, id_vars=[TIMESTAMP_COL], var_name='source', value_name='watts')
    ax = sns.lineplot(data=df, x=TIMESTAMP_COL, y='watts', hue='source')
    title = '{} {}'.format(extractor_name, energy_source)
    ax.set_title(title)
    figname = "extract_result_{}_{}".format(extractor_name, energy_source)
    plt.tight_layout()
    fig.savefig(_fig_filename(figname, save_path=save_path))

def plot_extract_results():
    for extractor in test_extractors:
        extractor_name = extractor.__class__.__name__
        extractor_results = get_extract_results(extractor_name, node_level=True)
        for feature_group, result in extractor_results.items():
            plot_extract_result(extractor_name, feature_group, result)
            plot_power_cols(extractor_name, result)

if __name__ == '__main__':
    plot_extract_results()