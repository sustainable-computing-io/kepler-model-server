from sklearn.metrics import mean_absolute_error

import os
import sys

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

from util import save_pkl, load_pkl, load_remote_pkl

from . import Trainer

model_class = "scikit"

def get_save_path(model_filepath):
    return "/".join(model_filepath.split("/")[0:-1])

class ScikitTrainer(Trainer):
    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name, scaler_type="minmax"):
        self.is_standard_scaler = scaler_type == "standard"
        super(ScikitTrainer, self).__init__(model_class, energy_components, feature_group, energy_source, node_level, pipeline_name, scaler_type=scaler_type)
        self.fe_files = []
 
    def train(self, node_type, component, X_values, y_values):
        if hasattr(self, 'fe'):
            for index in range(len(self.fe)):
                X_values = self.fe[index].fit_transform(X_values)
        model = self.node_models[node_type][component]
        model.fit(X_values, y_values)

    def save_checkpoint(self, model, filepath):
        if hasattr(self, 'fe'):
            save_path = get_save_path(filepath)
            for index in range(len(self.fe)):
                save_pkl(save_path, self.fe_files[index], self.fe[index])
        save_pkl("", filepath, model)

    def load_local_checkpoint(self, filepath):
        if hasattr(self, 'fe_files'):
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