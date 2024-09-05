from sklearn.linear_model import SGDRegressor

from kepler_model.train.trainer.scikit import ScikitTrainer


class SGDRegressorTrainer(ScikitTrainer):
    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name):
        super(SGDRegressorTrainer, self).__init__(energy_components, feature_group, energy_source, node_level, pipeline_name=pipeline_name)
        self.fe_files = []

    def init_model(self):
        return SGDRegressor(max_iter=1000)
