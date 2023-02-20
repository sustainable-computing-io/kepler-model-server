import os
import sys
from typing import List

train_path = os.path.join(os.path.dirname(__file__), '../../')
prom_path = os.path.join(os.path.dirname(__file__), '../../../prom')
sys.path.append(train_path)
sys.path.append(prom_path)

from pipeline import TrainPipeline
from train_types import FeatureGroup, FeatureGroups, CORE_COMPONENT, DRAM_COMPONENT, ModelOutputType
from pipe_util import XGBoostRegressionModelGenerationPipeline


MODEL_CLASS = 'xgboost'
energy_source = 'rapl'
energy_components = ['package', 'dram']

class XGBoostRegressionCompDynPipeline(TrainPipeline):

    def __init__(self) -> None:
        model_name = XGBoostRegressionCompDynPipeline.__name__
        model_file = model_name + ".model"
        self.model = None
        super().__init__(model_name, MODEL_CLASS, model_file, FeatureGroups[FeatureGroup.CgroupOnly], ModelOutputType.DynComponentPower)


    def initialize_model(self) -> None:
        self.model = XGBoostRegressionModelGenerationPipeline(self.features, ['package_joules'], '', self.model_name)


    def train(self, prom_client):
        pass

    def _get_model_path(self, model_type):
        return os.path.join(self.save_path, model_type)


    def predict(self, x_values: List[List[float]]):
        
        pass