import os
import sys
from typing import List, Tuple, Dict, Any
import pandas as pd

train_path = os.path.join(os.path.dirname(__file__), '../../')
prom_path = os.path.join(os.path.dirname(__file__), '../../../prom')
sys.path.append(train_path)
sys.path.append(prom_path)

from pipeline import TrainPipeline
from train_types import FeatureGroup, FeatureGroups, PACKAGE_LABEL, ModelOutputType, XGBoostRegressionTrainType
from pipe_util import XGBoostRegressionModelGenerationPipeline
from prom.query import PrometheusClient
from extractor import DefaultExtractor

# Currently Cgroup Metrics are not exported
class XGBoostRegressionCompDynPipeline(TrainPipeline):

    def __init__(self, train_type: XGBoostRegressionTrainType, save_location: str) -> None:
        self.train_type = train_type
        self.save_location = save_location
        model_name = XGBoostRegressionCompDynPipeline.__name__ + "_" + self.train_type.name
        model_file = model_name + ".model"
        self.model = None
        model_class = 'xgboost'
        self.energy_source = 'rapl'
        self.energy_components = ['package', 'dram']
        features = FeatureGroups[FeatureGroup.CgroupOnly]
        model_output = ModelOutputType.DynComponentPower
        self.labels = PACKAGE_LABEL
        super().__init__(model_name, model_class, model_file, features, model_output)
        self.initialize_relevant_models()

    # This pipeline is responsible for initializing one or more needed models
    # (Can vary depending on variables like isolator)
    def initialize_relevant_models(self) -> None:
        self.model = XGBoostRegressionModelGenerationPipeline(self.features, self.labels, self.save_location, self.model_name)

    def _generate_clean_model_training_data(self, extracted_data: pd.DataFrame) -> pd.DataFrame:
        # Merge Package data columns
        # Retrieve columns that end in "package_power"
        cloned_extracted_data = extracted_data.copy()
        # Column names should always be a string
        package_power_dataset = pd.DataFrame()
        for _, col_name in enumerate(cloned_extracted_data):
            if col_name.endswith("_package_power"):
                package_power_dataset[col_name] = cloned_extracted_data[col_name]
        # Sum to form total_package_power
        package_power_dataset['total_package_power'] = package_power_dataset.sum(axis=1)
        # Append total_package_power to cloned_extracted_data
        cloned_extracted_data['total_package_power'] = package_power_dataset['total_package_power']
        # Return cloned_extracted_data
        return cloned_extracted_data

    def train(self, prom_client: PrometheusClient) -> None:
        prom_client.query()
        results = prom_client.snapshot_query_result()
        # results can be used directly by extractor.py
        extractor = DefaultExtractor()

        extracted_data = extractor.extract(results, self.energy_components, self.feature_group.name, self.energy_source)
        if extracted_data is not None:
            print(extracted_data.head())
            clean_df = self._generate_clean_model_training_data(extracted_data)
            self.model.train(self.train_type, clean_df)
        else:
            raise Exception("extractor failed")

 
    def predict(self, list_of_features: List[List[float]]) -> Tuple[List[float], Dict[Any, Any]]:
        return self.model.predict(list_of_features)
        