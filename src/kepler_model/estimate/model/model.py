import json
import logging

import pandas as pd

from kepler_model.estimate.model.curvefit_model import CurveFitModelEstimator
from kepler_model.estimate.model.scikit_model import ScikitModelEstimator
from kepler_model.estimate.model.xgboost_model import XgboostModelEstimator
from kepler_model.util.config import download_path
from kepler_model.util.loader import get_download_output_path, load_metadata
from kepler_model.util.prom_types import valid_container_query

# from keras_model import KerasModelEstimator

logger = logging.getLogger(__name__)

# model wrapper
MODELCLASS = {
    "scikit": ScikitModelEstimator,
    "xgboost": XgboostModelEstimator,
    "curvefit": CurveFitModelEstimator,
    # 'keras': KerasModelEstimator,
}


def default_predicted_col_func(energy_component):
    return f"default_{energy_component}_power"


def default_idle_predicted_col_func(energy_component):
    return f"default_idle_{energy_component}_power"


def get_background_containers(idle_data):
    return pd.unique(idle_data[valid_container_query]["container_name"])


def get_label_power_colname(energy_component):
    return f"node_{energy_component}_power"


def get_predicted_power_colname(energy_component):
    return f"predicted_container_{energy_component}_power"


def get_predicted_background_power_colname(energy_component):
    return f"predicted_container_{energy_component}_background_power"


def get_dynamic_power_colname(energy_component):
    return f"container_{energy_component}_dynamic_power"


def get_predicted_dynamic_power_colname(energy_component):
    return f"predicted_container_{energy_component}_dynamic_power"


def get_predicted_dynamic_background_power_colname(energy_component):
    return f"predicted_container_{energy_component}_dynamic_background_power"


def get_reconstructed_power_colname(energy_component):
    return f"{energy_component}_reconstructed_power"


class Model:
    def __init__(
        self,
        model_path,
        model_class,
        model_name,
        output_type,
        model_file,
        features,
        fe_files=[],
        mae=None,
        mse=None,
        mape=None,
        mae_val=None,
        mse_val=None,
        abs_model=None,
        abs_mae=None,
        abs_mae_val=None,
        abs_mse=None,
        abs_mse_val=None,
        abs_max_corr=None,
        reconstructed_mae=None,
        reconstructed_mse=None,
        avg_mae=None,
        **kwargs,
    ):
        self.model_name = model_name
        self.trainer_name = model_name.split("_")[0]
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
            if attrb == "features":
                if not self.feature_check(val):
                    return False
            elif not hasattr(self, attrb) or getattr(self, attrb) is None:
                self.print_log(f"{self.model_name} has no {attrb}")
            else:
                cmp_val = getattr(self, attrb)
                val = float(val)
                if attrb == "abs_max_corr":  # higher is better
                    valid = cmp_val >= val
                else:  # lower is better
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
            self.print_log("Prediction error:" + msg)
            return None, None
        if hasattr(predicted_power_map, "items"):
            for energy_component, predicted_power in predicted_power_map.items():
                colname = predicted_col_func(energy_component)
                data_with_prediction[colname] = predicted_power
        else:
            # single list
            colname = predicted_col_func("platform")
            data_with_prediction[colname] = predicted_power
        return predicted_power_map, data_with_prediction

    def print_log(self, message):
        print(f"{self.model_name} model: {message}")

    def append_idle_prediction(self, data, predicted_col_func=default_idle_predicted_col_func):
        idle_data = data.copy()
        features = self.estimator.features
        idle_data[features] = 0
        return self.append_prediction(idle_data, predicted_col_func)


def load_model(model_path):
    metadata = load_metadata(model_path)
    if not metadata:
        logger.warning(f"no metadata in {model_path}")
        return None

    metadata["model_path"] = model_path
    metadata_str = json.dumps(metadata)
    try:
        model = json.loads(metadata_str, object_hook=lambda d: Model(**d))
        return model
    except Exception as e:
        logger.error(f"fail to load: {model_path} - {e}")
        return None


# download model folder has no subfolder of energy source and feature group because it has been already determined by model request
def load_downloaded_model(energy_source, output_type):
    model_path = get_download_output_path(download_path, energy_source, output_type)
    return load_model(model_path)
