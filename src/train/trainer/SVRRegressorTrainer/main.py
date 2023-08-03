from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

import os
import sys
trainer_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(trainer_path)

common_node_type = 1
from trainer.scikit import ScikitTrainer

class SVRRegressorTrainer(ScikitTrainer):
    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name):
        super(SVRRegressorTrainer, self).__init__(energy_components, feature_group, energy_source, node_level, scaler_type="standard", pipeline_name=pipeline_name)
        self.fe_files = []
    
    def init_model(self):
        print("scaler:", self.node_scalers.keys())
        return make_pipeline(self.node_scalers[common_node_type], SVR(C=1.0, epsilon=0.2))