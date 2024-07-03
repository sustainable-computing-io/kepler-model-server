import os
import sys

from sklearn.ensemble import GradientBoostingRegressor

trainer_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(trainer_path)

from trainer.scikit import ScikitTrainer

model_class = "scikit"


class GradientBoostingRegressorTrainer(ScikitTrainer):
    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name):
        super(GradientBoostingRegressorTrainer, self).__init__(
            energy_components, feature_group, energy_source, node_level, pipeline_name=pipeline_name
        )
        self.fe_files = []

    def init_model(self):
        return GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
