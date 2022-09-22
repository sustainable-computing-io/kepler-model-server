import os
import sys

train_path = os.path.join(os.path.dirname(__file__), '../../')
prom_path = os.path.join(os.path.dirname(__file__), '../../../prom')
sys.path.append(train_path)
sys.path.append(prom_path)

import json

from pipeline import TrainPipeline
from train_types import FeatureGroup, FeatureGroups, CORE_COMPONENT, DRAM_COMPONENT, ModelOutputType
from pipe_util import train_model_given_data_and_type, create_prometheus_core_dataset, create_prometheus_dram_dataset
from pipe_util import generate_core_regression_model, generate_dram_regression_model
from pipe_util import return_model_weights
from keras.models import load_model
from keras import backend as K

from prom.query import NODE_STAT_QUERY

MODEL_CLASS = 'keras'

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

class KerasCompWeightFullPipeline(TrainPipeline):
    def __init__(self):
        self.mse = None
        self.mse_val = None
        self.mae = None
        self.mae_val = None
        model_name = KerasCompWeightFullPipeline.__name__
        model_file = model_name + ".json"

        super(KerasCompWeightFullPipeline, self).__init__(model_name, MODEL_CLASS, model_file, FeatureGroups[FeatureGroup.Full], ModelOutputType.AbsComponentModelWeight)
        self.fe_files = []

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

        weights = dict()
        weights[CORE_COMPONENT] = return_model_weights(self._get_model_path(CORE_COMPONENT), CORE_COMPONENT)
        weights[DRAM_COMPONENT] = return_model_weights(self._get_model_path(DRAM_COMPONENT), DRAM_COMPONENT)
            
        with open(self._get_model_path(self.model_file), 'w') as f:
            obj = json.dumps(weights)
            f.write(obj)

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

    def predict(self, x_values):
        # TO-DO
        pass
