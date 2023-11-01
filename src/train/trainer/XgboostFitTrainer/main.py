from sklearn.model_selection import RepeatedKFold, cross_val_score

import os
import sys
trainer_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(trainer_path)

from train.trainer.xgboost_interface import XgboostTrainer

class XgboostFitTrainer(XgboostTrainer):
    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name):
        super(XgboostFitTrainer, self).__init__(energy_components, feature_group, energy_source, node_level, pipeline_name=pipeline_name)
        self.fe_files = []
    
    def _train(self, node_type, component, X_values, y_values):
        model = self.node_models[node_type][component]
        if model.__sklearn_is_fitted__():
            self.node_models[node_type][component].fit(X_values, y_values,  xgb_model=model)
        else:
            self.node_models[node_type][component].fit(X_values, y_values)
