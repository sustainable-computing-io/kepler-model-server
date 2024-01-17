import os
import sys

import pandas as pd

import json

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)
cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)

from loader import load_metadata, get_download_output_path
from config import download_path
from prom_types import TIMESTAMP_COL, valid_container_query

from scikit_model import ScikitModel
from xgboost_model import XgboostModel
from curvefit_model import CurveFitModel
# from keras_model import KerasModel

# model wrapper
MODELCLASS = {
    'scikit': ScikitModel,
    'xgboost': XgboostModel,
    'curvefit': CurveFitModel
    # 'keras': KerasModel,
}

def default_predicted_col_func(energy_component):
    return "default_{}_power".format(energy_component)

def get_background_containers(idle_data):
    return pd.unique(idle_data[valid_container_query]["container_name"])

def get_label_power_colname(energy_component):
    return "node_{}_power".format(energy_component)

def get_predicted_power_colname(energy_component):
    return "predicted_container_{}_power".format(energy_component)

def get_predicted_background_power_colname(energy_component):
    return "predicted_container_{}_background_power".format(energy_component)

def get_dynamic_power_colname(energy_component):
    return "container_{}_dynamic_power".format(energy_component)

def get_predicted_dynamic_power_colname(energy_component):
    return "predicted_container_{}_dynamic_power".format(energy_component)

def get_predicted_dynamic_background_power_colname(energy_component):
    return "predicted_container_{}_dynamic_background_power".format(energy_component)

def get_reconstructed_power_colname(energy_component):
    return "{}_reconstructed_power".format(energy_component)


class Model():
    def __init__(self, model_path, model_class, model_name, output_type, model_file, features, fe_files=[],\
            mae=None, mse=None, mape=None, mae_val=None, mse_val=None, \
            abs_model=None, abs_mae=None, abs_mae_val=None, abs_mse=None, abs_mse_val=None, abs_max_corr=None, \
            reconstructed_mae=None, reconstructed_mse=None, avg_mae=None, **kwargs):
        self.model_name = model_name
        self.estimator = MODELCLASS[model_class](model_path, model_name, output_type, model_file, features, fe_files)
        self.mae = mae
        self.mape = mape
        self.mae_val = mae_val
        self.mse = mse
        self.mse_val = mse_val
        self.abs_model = abs_model
        self.abs_mae = abs_mae
        self.abs_mae_val = abs_mae_val
        self.abs_mse = abs_mse
        self.abs_mse_val = abs_mse_val
        self.abs_max_corr = abs_max_corr
        self.reconstructed_mae = reconstructed_mae
        self.reconstructed_mse = reconstructed_mse
        self.avg_mae = avg_mae

    def get_power(self, data):
        return self.estimator.get_power(data)

    def is_valid_model(self, filters):
        for attrb, val in filters.items():
            if attrb == 'features':
                if not self.feature_check(val):
                    return False
            else:
                if not hasattr(self, attrb) or getattr(self, attrb) is None:
                    self.print_log('{} has no {}'.format(self.model_name, attrb))
                else:
                    cmp_val = getattr(self, attrb)
                    val = float(val)
                    if attrb == 'abs_max_corr': # higher is better
                        valid = cmp_val >= val
                    else: # lower is better
                        valid = cmp_val <= val
                    if not valid:
                        return False
        return True

    def feature_check(self, features):
        invalid_features = [f for f in self.estimator.features if f not in features]
        return len(invalid_features) == 0
    
    def append_prediction(self, data, predicted_col_func=default_predicted_col_func):
        data_with_prediction = data.copy()
        predicted_power_map, msg = self.estimator.get_power(data)
        if len(predicted_power_map) == 0:
            self.print_log('Prediction error:' + msg)
            return None, None
        if hasattr(predicted_power_map, 'items'):
            for energy_component, predicted_power in predicted_power_map.items():
                colname = predicted_col_func(energy_component)
                data_with_prediction[colname] = predicted_power
        else:
            # single list
            colname = predicted_col_func('platform')
            data_with_prediction[colname] = predicted_power    
        return predicted_power_map, data_with_prediction
    
    def print_log(self, message):
        print("{} model: {}".format(self.model_name, message))
        
def load_model(model_path):
    metadata = load_metadata(model_path)
    if metadata is not None:
        metadata['model_path'] = model_path
        metadata_str = json.dumps(metadata)
        try: 
            model = json.loads(metadata_str, object_hook = lambda d : Model(**d))
            return model
        except Exception as e:
            print("fail to load: ", e)
            return None
    print("no metadata")
    return None

# download model folder has no subfolder of energy source and feature group because it has been already determined by model request
def load_downloaded_model(energy_source, output_type):
    model_path = get_download_output_path(download_path, energy_source, output_type)
    return load_model(model_path)
