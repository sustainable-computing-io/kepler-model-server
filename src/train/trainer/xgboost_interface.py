from sklearn.metrics import  mean_absolute_error
import  os
import  sys
import  xgboost as xgb
import  numpy as np
import  base64

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

from util import  save_pkl, load_pkl
from abc import  abstractmethod

from . import  Trainer

model_class = "xgboost"

def get_save_path(model_filepath):
    return "/".join(model_filepath.split("/")[0:-1])

def _json_filepath(filepath):
    if ".json" not in filepath:
        filepath += ".json"
    return filepath

class XgboostTrainer(Trainer):
    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name, scaler_type="maxabs"):
        super(XgboostTrainer, self).__init__(model_class, energy_components, feature_group, energy_source, node_level, pipeline_name, scaler_type=scaler_type)
        self.fe_files = []

    def init_model(self):
        return xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    
    @abstractmethod
    def _train(self, node_type, component, X_values, y_values):
        return NotImplemented

    def train(self, node_type, component, X_values, y_values):
        if hasattr(self, 'fe'):
            for index in range(len(self.fe)):
                X_values = self.fe[index].fit_transform(X_values)
        self._train(node_type, component, X_values, y_values)

    def save_checkpoint(self, model, filepath):
        filepath = _json_filepath(filepath)
        if hasattr(self, 'fe'):
            save_path = get_save_path(filepath)
            for index in range(len(self.fe)):
                save_pkl(save_path, self.fe_files[index], self.fe[index])
        model.save_model(filepath)

    def load_local_checkpoint(self, filepath):
        filepath = _json_filepath(filepath)
        if hasattr(self, 'fe_files'):
            save_path = get_save_path(filepath)
            for index in range(len(self.fe_files)):
                loaded_fe = load_pkl(save_path, self.fe_files[index])
                if loaded_fe is not None:
                    self.fe[index] = loaded_fe
        loaded_model = None
        if os.path.exists(filepath):
            loaded_model = self.init_model()
            loaded_model.load_model(filepath)
        return loaded_model, loaded_model is not None

    #TODO
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
        filepath = os.path.join(component_save_path, self.component_model_filename(component))
        self.save_checkpoint(model, filepath)

    def component_model_filename(self, component):
        return component + ".json"
    
    
    def get_weight_dict(self, node_type):
        weight_dict = dict()
        scaler = self.node_scalers[node_type]
        for component in self.energy_components:
            checkpoint_filename = _json_filepath(self._checkpoint_filename(component, node_type))
            filename = os.path.join(self.checkpoint_toppath, checkpoint_filename)
            if not os.path.exists(filename):
                self.print_log("cannot get checkpoint file (in json) for xgboost")
                return
            with open(filename, "r") as f:
                contents = f.read()
            weight_dict[component] = {
                "All_Weights": {
                        "Categorical_Variables": dict(),
                        "Numerical_Variables": {self.features[i]: 
                                                {"scale": scaler.scale_[i]} for i in range(len(self.features))},
                        "XGboost_Weights": base64.b64encode(contents.encode('utf-8')).decode('utf-8')
                }
            }
        return weight_dict