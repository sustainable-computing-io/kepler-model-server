from random import random
from pipeline import TrainPipeline
from train_types import FeatureGroup, FeatureGroups
from kepler_model_trainer import train_model_given_data_and_type, create_prometheus_core_dataset, create_prometheus_dram_dataset, coeff_determination
from kepler_model_trainer import generate_core_regression_model, generate_dram_regression_model
from kepler_model_trainer import dram_model_labels, core_model_labels
from kepler_model_trainer import merge_model
from keras.models import load_model
import pickle
import tensorflow as tf

import os
import numpy as np

MODEL_NAME = 'KerasLR_Full'
MODEL_CLASS = 'keras'
MODEL_FILENAME = MODEL_NAME + '.h5'
FE_FILE = 'merge.pkl'

CORE_MODEL_TYPE = 'core'
DRAM_MODEL_TYPE = 'dram'

class KerasFullPipelineFeatureTransformer():
    def __init__(self, features, dram_model_labels, core_model_labels):
        self.features = features
        self.core_numerical_indexes = [self.features.index(label) for label in core_model_labels['numerical_labels']]
        self.dram_numerical_indexes = [self.features.index(label) for label in dram_model_labels['numerical_labels']]
        self.core_categorical_indexes = [self.features.index(label) for label in core_model_labels['categorical_string_labels']]
        self.dram_categorical_indexes = [self.features.index(label) for label in core_model_labels['categorical_string_labels']]

    def transform(self, x_values):
        inputs = []
        for core_index in self.core_numerical_indexes:
            core_values = x_values[:,core_index:core_index+1].astype(np.float32)
            inputs.append(tf.cast(core_values, tf.float32))
        for core_index in self.core_categorical_indexes:
            inputs.append(tf.cast(x_values[:,core_index:core_index+1], tf.string))
        for dram_index in self.dram_numerical_indexes:
            dram_values = x_values[:,dram_index:dram_index+1].astype(np.float32)
            inputs.append(tf.cast(dram_values, tf.float32))
        for dram_index in self.dram_categorical_indexes:
            inputs.append(tf.cast(x_values[:,dram_index:dram_index+1], tf.string))
        return inputs

class KerasFullPipeline(TrainPipeline):
    def __init__(self):
        self.mse = None
        self.mse_val = None
        self.mae = None
        self.mae_val = None

        super(KerasFullPipeline, self).__init__(MODEL_NAME, MODEL_CLASS, MODEL_FILENAME, FeatureGroups[FeatureGroup.Full])
        self.fe = KerasFullPipelineFeatureTransformer(self.features, dram_model_labels, core_model_labels)
        fe_file_path = os.path.join(self.save_path, FE_FILE)
        with open(fe_file_path, 'wb') as f:
            pickle.dump(self.fe, f)
            self.fe_files = [FE_FILE]

    def train(self, pod_stat_data, node_stat_data, freq_data, pkg_data):
        if int(node_stat_data['node_curr_energy_in_pkg_joule'].sum()) == 0:
            # cannot train with package-level info
            return
        core_train, core_val, core_test = create_prometheus_core_dataset(node_stat_data['cpu_architecture'], node_stat_data['curr_cpu_cycles'], node_stat_data['curr_cpu_instr'], node_stat_data['curr_cpu_time'], node_stat_data['node_curr_energy_in_core_joule'])
        dram_train, dram_val, dram_test = create_prometheus_dram_dataset(node_stat_data['cpu_architecture'], node_stat_data['curr_cache_miss'], node_stat_data['container_memory_working_set_bytes'], node_stat_data['node_curr_energy_in_dram_joule'])
        core_model = self.load_model(core_train, CORE_MODEL_TYPE)
        dram_model = self.load_model(dram_train, DRAM_MODEL_TYPE)
        core_model, _, _, core_mae_metric = train_model_given_data_and_type(core_model, core_train, core_val, core_test, CORE_MODEL_TYPE)
        dram_model, _, _, dram_mae_metric = train_model_given_data_and_type(dram_model, dram_train, dram_val, dram_test, DRAM_MODEL_TYPE)
        core_model.save(self._get_model_path(CORE_MODEL_TYPE), include_optimizer=True)
        dram_model.save(self._get_model_path(DRAM_MODEL_TYPE), include_optimizer=True)
        merged_model = merge_model(core_model, dram_model)
        merged_model.save(self._get_model_path(MODEL_NAME), include_optimizer=True)
        metadata = dict()
        metadata['mae'] = core_mae_metric + dram_mae_metric
        self.update_metadata(metadata)

    def _get_model_path(self, model_type):
        return os.path.join(self.save_path, model_type + ".h5")

    def load_model(self, train_dataset, model_type):
        filepath = self._get_model_path(model_type)
        model_exists = os.path.exists(filepath)
        if model_exists:
            new_model = load_model(filepath, custom_objects={'coeff_determination': coeff_determination})
        elif model_type == CORE_MODEL_TYPE:
            new_model = generate_core_regression_model(train_dataset)
        elif model_type == DRAM_MODEL_TYPE:
            new_model = generate_dram_regression_model(train_dataset)
        else:
            "wrong type: " + model_type
        return new_model

    def predict(self, x_values):
        print(x_values.shape)
        fe_filepath = os.path.join(self.save_path, FE_FILE)
        fe_item = pickle.load(open(fe_filepath, 'rb'))
        transformed_values = fe_item.transform(x_values)
        model_filepath = os.path.join(self.save_path, MODEL_FILENAME)
        merged_model = load_model(model_filepath, compile=False)
        result = merged_model.predict(transformed_values)
        print(result)


if __name__ == '__main__':
    from train_types import WORKLOAD_FEATURES, SYSTEM_FEATURES, CATEGORICAL_LABEL_TO_VOCAB, NODE_STAT_POWER_LABEL
    import pandas as pd
    import numpy as np
    item = dict()

    for f in WORKLOAD_FEATURES:
        item[f] = random()
    for f in SYSTEM_FEATURES:
        if f in CATEGORICAL_LABEL_TO_VOCAB:
            item[f] = CATEGORICAL_LABEL_TO_VOCAB[f][0]
        else:
            item[f] = random()
    for l in NODE_STAT_POWER_LABEL:
        item[l] = random()*1000

    data_items = [item]
    node_stat_data = pd.DataFrame(data_items)
    for f in WORKLOAD_FEATURES:
        node_stat_data[f] = node_stat_data[f].astype(np.float32)
    for f in SYSTEM_FEATURES:
        if f not in CATEGORICAL_LABEL_TO_VOCAB:
            node_stat_data[f] = node_stat_data[f].astype(np.float32)

    node_stat_data = pd.concat([node_stat_data]*10, ignore_index=True)
    pipeline = KerasFullPipeline()
    pipeline.train(None, node_stat_data, None, None)
    pipeline.predict(node_stat_data[pipeline.features].values)