import math

import numpy as np

from kepler_model.train.trainer.curvefit import CurveFitModel, CurveFitTrainer


def p0_func(x, y):
    a = (y.max() - y.min()) // math.e  # scale value
    b = 1  # start from linear
    c = y.min() - a  # initial offset
    return [a, b, c]


def expo_func(x, a, b, c):
    y = a * np.exp(b * x) + c
    return y


class ExponentialRegressionTrainer(CurveFitTrainer):
    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name):
        super(ExponentialRegressionTrainer, self).__init__(energy_components, feature_group, energy_source, node_level, pipeline_name=pipeline_name)
        self.fe_files = []

    def init_model(self):
        return CurveFitModel(expo_func, p0_func=p0_func)

