from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from urllib.request import urlopen

import os
import sys

from . import Trainer

model_class = "scikit"

class ScikitTrainer(Trainer):
    def __init__(self, profiles, energy_components, feature_group, energy_source, node_level, scaler_type="minmax"):
        self.is_standard_scaler = scaler_type == "standard"
        super(ScikitTrainer, self).__init__(profiles, model_class, energy_components, feature_group, energy_source, node_level, scaler_type=scaler_type)
        self.fe_files = []
 
    def train(self, node_type, component, X_values, y_values):
        model = self.node_models[node_type][component]
        model.fit(X_values, y_values)

    def save_checkpoint(self, model, filepath):
        filepath += ".pkl"
        joblib.dump(model, filepath)

    def load_local_checkpoint(self, filepath):
        filepath += ".pkl"
        try:
            loaded_model = joblib.load(filepath)
            return loaded_model, True
        except:
            return None, False

    def load_remote_checkpoint(self, url_path):
        url_path += ".pkl"
        try:        
            response = urlopen(url_path)
            loaded_model = joblib.load(response)
            return loaded_model, True
        except:
            return None, False

    def should_archive(self, node_type):
        return True

    def get_basic_metadata(self, node_type):
        return dict()

    def get_mae(self, node_type, component, X_test, y_test):
        model = self.node_models[node_type][component]
        predicted_values = model.predict(X_test)
        mae = mean_absolute_error(y_test,predicted_values)
        return mae

    def save_model(self, component_save_path, node_type, component):
        model = self.node_models[node_type][component]
        filepath = os.path.join(component_save_path, component)
        self.save_checkpoint(model, filepath)

    def component_model_filename(self, component):
        return component + ".pkl"

    def get_weight_dict(self, node_type):
        if not self.is_standard_scaler:
            # cannot get weight dict
            return None
        weight_dict = dict()

        for component, model in self.node_models[node_type].items():
            scaler = self.node_scalers[node_type]
            if not hasattr(model, "intercept_") or not hasattr(model, "coef_") or len(model.coef_) != len(self.features) or len(model.intercept_) != 1:
                return None
            else:
                weight_dict[component] = {
                    "All_Weights": {
                        "Bias_Weight": model.intercept_[0],
                        "Categorical_Variables": dict(),
                        "Numerical_Variables": {self.features[i]: 
                                                {"mean": scaler.mean_[i], 
                                                "variance": scaler.var_[i], 
                                                "weight": model.coef_[i], 
                                                }
                                                for i in range(len(self.features))},
                    }
                }
        return weight_dict