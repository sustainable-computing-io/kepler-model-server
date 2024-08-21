from sklearn.svm import SVR

from kepler_model.train.trainer.scikit import ScikitTrainer

common_node_type = 1


class SVRRegressorTrainer(ScikitTrainer):
    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name):
        super(SVRRegressorTrainer, self).__init__(energy_components, feature_group, energy_source, node_level, pipeline_name=pipeline_name)
        self.fe_files = []

    def init_model(self):
        return SVR(C=1.0, epsilon=0.2)

