import os
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error

util_path = os.path.join(os.path.dirname(__file__), "..", "..", "util")
sys.path.append(util_path)

from util import load_pkl, save_pkl

from . import Trainer

model_class = "scikit"


def get_save_path(model_filepath):
    return "/".join(model_filepath.split("/")[0:-1])


class ScikitTrainer(Trainer):
    def __init__(
        self, energy_components, feature_group, energy_source, node_level, pipeline_name, scaler_type="maxabs"
    ):
        super(ScikitTrainer, self).__init__(
            model_class,
            energy_components,
            feature_group,
            energy_source,
            node_level,
            pipeline_name,
            scaler_type=scaler_type,
        )
        self.fe_files = []

    def train(self, node_type, component, X_values, y_values):
        if hasattr(self, "fe"):
            for index in range(len(self.fe)):
                X_values = self.fe[index].fit_transform(X_values)
        model = self.node_models[node_type][component]
        model.fit(X_values, y_values)

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
        non_zero_predicted_values = np.array(
            [predicted_values[i] for i in range(len(predicted_values)) if y_test[i] > 0]
        )
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
            if (
                not hasattr(model, "intercept_")
                or not hasattr(model, "coef_")
                or len(model.coef_) != len(self.features)
                or (hasattr(model.intercept_, "__len__") and len(model.intercept_) != 1)
            ):
                return None
            else:
                if isinstance(model.intercept_, np.float64):
                    intercept = model.intercept_
                elif hasattr(model.intercept_, "__len__"):
                    intercept = model.intercept_[0]
                else:
                    # no valid intercept
                    return None

                # TODO: remove the mean and variance variables after updating the Kepler code
                weight_dict[component] = {
                    "All_Weights": {
                        "Bias_Weight": intercept,
                        "Categorical_Variables": dict(),
                        "Numerical_Variables": {
                            self.features[i]: {
                                "scale": scaler.scale_[i],
                                "weight": model.coef_[i],
                            }
                            for i in range(len(self.features))
                        },
                    }
                }
        return weight_dict
