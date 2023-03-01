from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from urllib.request import urlopen

import os
import sys
trainer_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(trainer_path)

from trainer.scikit import ScikitTrainer

model_class = "scikit"

class GradientBoostingRegressorTrainer(ScikitTrainer):
    def __init__(self, profiles, energy_components, feature_group, energy_source, node_level):
        super(GradientBoostingRegressorTrainer, self).__init__(profiles, energy_components, feature_group, energy_source, node_level)
        self.fe_files = []
    
    def init_model(self):
        return GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
