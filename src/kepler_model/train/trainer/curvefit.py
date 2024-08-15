from sklearn.metrics import mean_absolute_error
from sklearn.exceptions import NotFittedError
import numpy as np
from scipy.optimize import curve_fit
import os

from kepler_model.util import save_pkl, load_pkl
from kepler_model.util.train_types import main_feature

from . import Trainer

model_class = "curvefit"


def get_save_path(model_filepath):
    return "/".join(model_filepath.split("/")[0:-1])


class CurveFitModel:
    def __init__(self, fit_func, p0_func=None):
        self.fit_func = fit_func
        self.popt = None
        self.pcov = None
        self.feature_index = None
        self.p0_func = p0_func

    def set_feature_index(self, feature_index):
        self.feature_index = feature_index

    def _x_values(self, X_values):
        return np.array(X_values[:, self.feature_index]).flatten()

    def fit(self, X_values, y_values):
        flatten_x = self._x_values(X_values)
        flatten_y = np.array(y_values).flatten()
        if self.p0_func is not None:
            self.popt, self.pcov = curve_fit(self.fit_func, flatten_x, flatten_y, p0=self.p0_func(flatten_x, flatten_y), maxfev=30000)
        else:
            self.popt, self.pcov = curve_fit(self.fit_func, flatten_x, flatten_y, maxfev=30000)

    def predict(self, X_values):
        if self.popt is None:
            raise NotFittedError("Model must be fit first")
        flatten_x = self._x_values(X_values)
        return np.array(self.fit_func(flatten_x, *self.popt))


# curvefit will focus on only single feature. default is the first feature in the feature group.
class CurveFitTrainer(Trainer):
    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name, scaler_type="maxabs"):
        super(CurveFitTrainer, self).__init__(model_class, energy_components, feature_group, energy_source, node_level, pipeline_name, scaler_type=scaler_type)
        self.fe_files = []

    def train(self, node_type, component, X_values, y_values):
        try:
            if hasattr(self, "fe"):
                for index in range(len(self.fe)):
                    X_values = self.fe[index].fit_transform(X_values)
            model = self.node_models[node_type][component]
            if component == "package":
                dram_index = main_feature(self.feature_group_name, "dram")
                if model.feature_index != dram_index:
                    dram_values = np.array(X_values[:, dram_index]).flatten()
                    zero_dram_indices = [i for i in dram_values if i < 0.1]
                    X_values = [list(row) for i, row in enumerate(X_values) if i not in zero_dram_indices]
                    y_values = [row for i, row in enumerate(y_values) if i not in zero_dram_indices]
                    X_values = np.array(X_values)
            model.fit(X_values, y_values)
        except Exception as err:
            print("Train error", err)
            import traceback

            traceback.print_exc()

    def save_checkpoint(self, model, filepath):
        if hasattr(self, "fe"):
            save_path = get_save_path(filepath)
            for index in range(len(self.fe)):
                save_pkl(save_path, self.fe_files[index], self.fe[index])
        save_pkl("", filepath, model)

    def load_local_checkpoint(self, filepath):
        if hasattr(self, "fe_files"):
            save_path = get_save_path(filepath)
            for index in range(len(self.fe_files)):
                loaded_fe = load_pkl(save_path, self.fe_files[index])
                if loaded_fe is not None:
                    self.fe[index] = loaded_fe
        loaded_model = load_pkl("", filepath)
        return loaded_model, loaded_model is not None

    def should_archive(self, node_type):
        return True

    def get_basic_metadata(self, node_type):
        return dict()

    def get_mae(self, node_type, component, X_test, y_test):
        predicted_values = self.predict(node_type, component, X_test, skip_preprocess=True)
        mae = mean_absolute_error(y_test, predicted_values)
        return mae

    def get_mape(self, node_type, component, X_test, y_test):
        y_test = list(y_test)
        predicted_values = self.predict(node_type, component, X_test, skip_preprocess=True)
        non_zero_predicted_values = np.array([predicted_values[i] for i in range(len(predicted_values)) if y_test[i] > 0])
        if len(non_zero_predicted_values) == 0:
            return -1
        non_zero_y_test = np.array([y for y in y_test if y > 0])
        absolute_percentage_errors = np.abs((non_zero_y_test - non_zero_predicted_values) / non_zero_y_test) * 100
        mape = np.mean(absolute_percentage_errors)
        return mape

    def save_model(self, component_save_path, node_type, component):
        model = self.node_models[node_type][component]
        filepath = os.path.join(component_save_path, component)
        self.save_checkpoint(model, filepath)

    def component_model_filename(self, component):
        return component + ".pkl"

    def get_weight_dict(self, node_type):
        weight_dict = dict()

        for component, model in self.node_models[node_type].items():
            scaler = self.node_scalers[node_type]
            weight_dict[component] = {"All_Weights": {"Categorical_Variables": dict(), "Numerical_Variables": {self.features[i]: {"scale": scaler.scale_[i]} for i in range(len(self.features))}, "CurveFit_Weights": list(model.popt)}}
        return weight_dict

