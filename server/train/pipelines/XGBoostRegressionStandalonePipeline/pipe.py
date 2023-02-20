import os
import sys
from typing import Optional
import logging

train_path = os.path.join(os.path.dirname(__file__), '../../')
prom_path = os.path.join(os.path.dirname(__file__), '../../../prom')
sys.path.append(train_path)
sys.path.append(prom_path)

import prom.query as q
import pipeline as p
import train_types as tt
import extractor as e
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

# CREATE SEPARATE UTIL TOOL IN PIPE_UTIL TO HANDLE ALL XGBOOSTREGRESSION RELATED TO MODEL SERVER! CERTAIN FIELDS ARE REQUIRED
# WHICH WILL BE PROVIDED BY A SEPARATE FUNCTION (SEPARATE FUNCTION? OR JUST STORED IN MODEL_SERVER.PY)
# ALSO SET UP FUNCTIONALITY TO PUSH MODEL TO KEPLER-MODELS
# THESE PIPES WILL JUST FEED THE DESIRED FIELDS TO THE UTIL FUNCTION AND WILL BE DIRECTLY
# USED IN MODEL_SERVER.PY (CALL DESIRED PIPELINE FOR UP TO DATE PREDICTION)
# USED IN ONLINE_TRAINER.PY TO TRAIN - INCLUDE FUNCTION THAT ALLOWS RETURNING OF MODEL?
# NEED TO CHECK HOW THIS WORKS
MODEL_CLASS = "xgboost"
energy_components = []
energy_source = 'rapl'

class XGBoostRegressionPipeline(p.TrainPipeline):
    def __init__(self) -> None:
        model_name = XGBoostRegressionPipeline.__name__
        model_file = model_name + ".json" # may require changes
        self.fe_files = []
        
        super().__init__(model_name, MODEL_CLASS, model_file, tt.FeatureGroups[tt.FeatureGroup.IRQOnly], tt.ModelOutputType.DynComponentModelWeight)

    def __extract_prom_data(self, prom_client: q.PrometheusClient) -> pd.DataFrame:
        prom_client.query()
        retrieved_snapshot = prom_client.snapshot_query_result()
        # TODO: retrieve desired feature names and label
        extractor = e.DefaultExtractor()
        retrieved_df = extractor.extract(retrieved_snapshot, energy_components, self.features, energy_source)
        return None
        # return two dataframes, one with features, one with the label

    def load_model(self, model_name) -> Optional[xgb.XGBRegressor]:
        # Require a save folder path
        pass
        
    def train(self, prom_client) -> None:
        dyn_comp_full_df = self.__extract_prom_data(prom_client)
        # dyn_comp_full_df contains features and label of desired model 
        # dyn_comp_full_df is assumed to be a regression problem as it extracts 
        if dyn_comp_full_df is None:
            return
        # important parameters to consider tuning:
        # n_estimators, max_depth, eta (learning rate)
        standalone_model = self.load_model(self.model_name)
        if standalone_model is None:
            # Generate new model
            standalone_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1) 
        

