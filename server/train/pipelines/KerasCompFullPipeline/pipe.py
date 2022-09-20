import os
import sys

train_path = os.path.join(os.path.dirname(__file__), '../../')
prom_path = os.path.join(os.path.dirname(__file__), '../../../prom')
sys.path.append(train_path)
sys.path.append(prom_path)

from pipeline import TrainPipeline
from train_types import FeatureGroup, FeatureGroups, CORE_COMPONENT, DRAM_COMPONENT, ModelOutputType
from pipe_util import train_model_given_data_and_type, create_prometheus_core_dataset, create_prometheus_dram_dataset
from pipe_util import generate_core_regression_model, generate_dram_regression_model, dram_model_labels, core_model_labels
from keras.models import load_model
from keras import backend as K
from transformer import KerasFullPipelineFeatureTransformer

import json
import pickle

from prom.query import NODE_STAT_QUERY

MODEL_CLASS = 'keras'

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

class KerasCompFullPipeline(TrainPipeline):
    def __init__(self):
        self.mse = None
        self.mse_val = None
        self.mae = None
        self.mae_val = None
        model_name = KerasCompFullPipeline.__name__
        model_file = model_name + ".json"

        super(KerasCompFullPipeline, self).__init__(model_name, MODEL_CLASS, model_file, FeatureGroups[FeatureGroup.Full], ModelOutputType.AbsComponentPower)
        self.fe_files = []

        self.model_file_dict = {CORE_COMPONENT: CORE_COMPONENT, DRAM_COMPONENT: DRAM_COMPONENT}
        self.model_features_dict = {CORE_COMPONENT: ['cpu_architecture', 'cpu_cycles', 'cpu_instr', 'cpu_time'], DRAM_COMPONENT: ['cpu_architecture', 'cache_miss', 'container_memory_working_set_bytes']}
        self.model_fe_file_dict = dict()
        self.prepare_fe(CORE_COMPONENT)
        self.prepare_fe(DRAM_COMPONENT)


    def prepare_fe(self, model_type):
        if model_type == CORE_COMPONENT:
            fe = KerasFullPipelineFeatureTransformer(self.model_features_dict[model_type], {}, core_model_labels)
        elif model_type == DRAM_COMPONENT:
            fe = KerasFullPipelineFeatureTransformer(self.model_features_dict[model_type], dram_model_labels, {})
        fe_filename = "{}_transform.pkl".format(model_type)
        fe_file_path = os.path.join(self.save_path, fe_filename)
        with open(fe_file_path, 'wb') as f:
            pickle.dump(fe, f)
            self.model_fe_file_dict[model_type] = [fe_filename]

    def train(self, prom_client):
        node_stat_data = prom_client.get_data(NODE_STAT_QUERY, None)
        if int(node_stat_data['energy_in_pkg_joule'].sum()) == 0:
            # cannot train with package-level info
            return
        core_train, core_val, core_test = create_prometheus_core_dataset(node_stat_data['cpu_architecture'], node_stat_data['cpu_cycles'], node_stat_data['cpu_instr'], node_stat_data['cpu_time'], node_stat_data['energy_in_core_joule'])
        dram_train, dram_val, dram_test = create_prometheus_dram_dataset(node_stat_data['cpu_architecture'], node_stat_data['cache_miss'], node_stat_data['container_memory_working_set_bytes'], node_stat_data['energy_in_dram_joule'])
        core_model = self.load_model(core_train, CORE_COMPONENT)
        dram_model = self.load_model(dram_train, DRAM_COMPONENT)
        core_model, _, _, core_mae_metric = train_model_given_data_and_type(core_model, core_train, core_val, core_test, CORE_COMPONENT)
        dram_model, _, _, dram_mae_metric = train_model_given_data_and_type(dram_model, dram_train, dram_val, dram_test, DRAM_COMPONENT)
        core_model.save(self._get_model_path(CORE_COMPONENT), save_format='tf', include_optimizer=True)
        dram_model.save(self._get_model_path(DRAM_COMPONENT), save_format='tf', include_optimizer=True)

        self.save_comp_model(self.model_file_dict, self.model_features_dict, self.model_fe_file_dict)

        metadata = dict()
        metadata['mae'] = core_mae_metric + dram_mae_metric
        self.update_metadata(metadata)

    def _get_model_path(self, model_type):
        return os.path.join(self.save_path, model_type)

    def load_model(self, train_dataset, model_type):
        filepath = self._get_model_path(model_type)
        model_exists = os.path.exists(filepath)
        if model_exists:
            new_model = load_model(filepath, custom_objects={'coeff_determination': coeff_determination})
        elif model_type == CORE_COMPONENT:
            new_model = generate_core_regression_model(train_dataset)
        elif model_type == DRAM_COMPONENT:
            new_model = generate_dram_regression_model(train_dataset)
        else:
            "wrong type: " + model_type
        return new_model

    def predict(self, data):
        results = dict()
        model_file = self._get_model_path(self.model_file)
        with open(model_file) as f:
            model_info = json.load(f)
        for comp, model_metadata in model_info.items():
            model = load_model(self._get_model_path(model_metadata['model_file']), custom_objects={'coeff_determination': coeff_determination})
            features = model_metadata['features']
            fe_files = model_metadata['fe_files']
            x_values = data[features].values
            for fe_filename in fe_files:
                fe_filepath = self._get_model_path(fe_filename)
                with open(fe_filepath, 'rb') as f:
                    fe = pickle.load(f)
                    x_values = fe.transform(x_values)
            results[comp] = model.predict(x_values)
        return results