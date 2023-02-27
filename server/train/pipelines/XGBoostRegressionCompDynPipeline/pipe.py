import os
import sys
from typing import List, Tuple, Dict, Any
import pandas as pd
import copy

train_path = os.path.join(os.path.dirname(__file__), '../../')
prom_path = os.path.join(os.path.dirname(__file__), '../../../prom')
sys.path.append(train_path)
sys.path.append(prom_path)

from train_types import FeatureGroup, FeatureGroups, EnergyComponentLabelGroups, EnergyComponentLabelGroup, ModelOutputType, XGBoostRegressionTrainType
from pipe_util import XGBoostRegressionModelGenerationPipeline
from prom.query import PrometheusClient
from extractor import DefaultExtractor


# Currently Cgroup Metrics are not exported
class XGBoostRegressionStandalonePipeline():

    def __init__(self, train_type: XGBoostRegressionTrainType, save_location: str, node_level: bool) -> None:
        self.model = None
        self.train_type = train_type
        self.model_class = 'xgboost'
        self.energy_source = 'rapl'
        self.feature_group = FeatureGroup.CounterOnly
        self.save_location = save_location
        self.energy_components_labels = EnergyComponentLabelGroups[EnergyComponentLabelGroup.PackageEnergyComponentOnly]
        self.features = FeatureGroups[self.feature_group]
        print(self.energy_components_labels)
        self.model_labels = ["total_package_power"]
        # Define key model names
        if node_level:
            self.model_name = XGBoostRegressionStandalonePipeline.__name__ + "_" + "Node_Level" + "_" + self.train_type.name
        else:
            self.model_name = XGBoostRegressionStandalonePipeline.__name__ + "_" + "Container_Level" + "_" + self.train_type.name
        self.node_level = node_level
        self.initialize_relevant_models()

    # This pipeline is responsible for initializing one or more needed models
    # (Can vary depending on variables like isolator)
    def initialize_relevant_models(self) -> None:
        # Generate models and store in dict
        self.model = XGBoostRegressionModelGenerationPipeline(self.features, self.model_labels, self.save_location, self.model_name)
        

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
        package_power_dataset[self.model_labels[0]] = package_power_dataset.sum(axis=1)
        # Append total_package_power to cloned_extracted_data
        cloned_extracted_data[self.model_labels[0]] = package_power_dataset[self.model_labels[0]]
        # Return cloned_extracted_data
        return cloned_extracted_data


    def train(self, prom_client: PrometheusClient) -> None:
        prom_client.query()
        results = prom_client.snapshot_query_result()
        # results can be used directly by extractor.py
        extractor = DefaultExtractor()
        # Train all models with extractor
        extracted_data = extractor.extract(results, self.energy_components_labels, self.feature_group.name, self.energy_source, node_level=self.node_level)

        if extracted_data is not None:
            clean_df = self._generate_clean_model_training_data(extracted_data)
            self.model.train(self.train_type, clean_df)
        else:
            raise Exception("extractor failed")
 
    # Accepts JSON Input with feature and corresponding prediction
    def predict(self, features_and_predictions: List[Dict[str,float]]) -> Tuple[List[float], Dict[Any, Any]]:
        # features Convert to List[List[float]]
        list_of_predictions = []
        for prediction in features_and_predictions:
            feature_values = []
            for feature in self.features:
                feature_values.append(prediction[feature])
            list_of_predictions.append(feature_values)
        return self.model.predict(list_of_predictions)