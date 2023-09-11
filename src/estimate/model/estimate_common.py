keras_enabled = True
import cpuinfo

cpu_info = cpuinfo.get_cpu_info()

# if 'flags' in cpu_info and 'avx' in cpu_info['flags']:
#     import keras
#     from keras import backend as K
# else:
#     print("AVX instructions are not available.")
#     keras_enabled = False

from sklearn.metrics import mean_squared_error, mean_absolute_error

import os
import sys

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

from loader import load_pkl, load_json

def is_component_model(model_file):
    return '.json' in model_file

def transform_and_predict(model, datapoint):
    msg = ""
    try:
        x_values = datapoint[model.features].values
        for fe in model.fe_list:
            if fe is None:
                continue
            x_values = fe.transform(x_values)
        y = model.model.predict(x_values).squeeze()
        y[y < 0] = 0
        y = y.tolist()
    except Exception as e:
        msg = '{}\n'.format(e)
        y = []
    return y, msg

def load_model_by_pickle(model_path, model_filename):
    return load_pkl(model_path, model_filename)

def coeff_determination(y_true, y_pred):
    if not keras_enabled:
        return None
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def load_model_by_keras(model_path, model_filename):
    model_file = os.path.join(model_path, model_filename)
    try:
        model = keras.models.load_model(model_file, custom_objects={'coeff_determination': coeff_determination})
    except Exception as e:
        print(e)
        return None
    return model

def load_model_by_json(model_path, model_filename):
    return load_json(model_path, model_filename)

def compute_error(predicted_power, actual_powers):
    mse = mean_squared_error(actual_powers, predicted_power)
    mae = mean_absolute_error(actual_powers, predicted_power)
    return mae, mse
