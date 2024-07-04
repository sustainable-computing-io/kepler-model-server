import os
import sys

from sklearn.linear_model import LinearRegression

trainer_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(trainer_path)

from trainer.scikit import ScikitTrainer

model_class = "scikit"


class LinearRegressionTrainer(ScikitTrainer):
    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name):
        super(LinearRegressionTrainer, self).__init__(
            energy_components, feature_group, energy_source, node_level, pipeline_name=pipeline_name
        )
        self.fe_files = []

    def init_model(self):
        return LinearRegression(positive=True)
