import numpy as np
from kepler_model.train.trainer.curvefit import CurveFitTrainer, CurveFitModel


def p0_func(x, y):
    A = y.max() - y.min()  # value range
    x0 = 0.5  # sigmoid mid point (as normalized value is in 0 to 1, start mid point = 0.5)
    k = A // np.std(y)  # growth rate (larger std, lower growth)
    off = y.min()  # initial offset
    return [A, x0, k, off]


def logi_func(x, A, x0, k, off):
    return A / (1 + np.exp(-k * (x - x0))) + off


class LogisticRegressionTrainer(CurveFitTrainer):
    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name):
        super(LogisticRegressionTrainer, self).__init__(energy_components, feature_group, energy_source, node_level, pipeline_name=pipeline_name)
        self.fe_files = []

    def init_model(self):
        return CurveFitModel(logi_func, p0_func=p0_func)
