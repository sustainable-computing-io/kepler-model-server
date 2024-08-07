from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import  PolynomialFeatures
from urllib.request import  urlopen

import  os
import  sys
trainer_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(trainer_path)

from trainer.scikit import  ScikitTrainer

poly_scaler_filename = "poly_scaler.pkl"

class PolynomialRegressionTrainer(ScikitTrainer):

    def __init__(self, energy_components, feature_group, energy_source, node_level, pipeline_name):
        super(PolynomialRegressionTrainer, self).__init__(energy_components, feature_group, energy_source, node_level, pipeline_name=pipeline_name)
        self.poly_scaler = PolynomialFeatures(degree=2)
        self.fe_files = [poly_scaler_filename]
        self.fe = [PolynomialFeatures(degree=2)]

    def init_model(self):
        return LinearRegression()