import os
import sys
trainer_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(trainer_path)

from trainer.curvefit import CurveFitTrainer, CurveFitModel

import numpy as np

def p0_func(x, y):
    a = y.max()-y.min()
    b = 1
    c = y.min()
    return [a, b, c]

def log_func(x, a, b, c):
    y = a*np.log(b*x+1) + c
    return y

class LogarithmicRegressionTrainer(CurveFitTrainer):

    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name):
        super(LogarithmicRegressionTrainer, self).__init__(energy_components, feature_group, energy_source, node_level, pipeline_name=pipeline_name)
        self.fe_files = []
    
    def init_model(self):
        return CurveFitModel(log_func, p0_func=p0_func)