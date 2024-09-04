from sklearn.neighbors import KNeighborsRegressor

from kepler_model.train.trainer.scikit import ScikitTrainer

model_class = "scikit"


class KNeighborsRegressorTrainer(ScikitTrainer):
    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name):
        super(KNeighborsRegressorTrainer, self).__init__(energy_components, feature_group, energy_source, node_level, pipeline_name=pipeline_name)
        self.fe_files = []

    def init_model(self):
        return KNeighborsRegressor(n_neighbors=6)
