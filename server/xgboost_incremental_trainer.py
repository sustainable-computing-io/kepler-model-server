import os
import sys
import time

#util_path = os.path.join(os.path.dirname(__file__), 'util')
train_path = os.path.join(os.path.dirname(__file__), 'train')
prom_path = os.path.join(os.path.dirname(__file__), 'prom')

#sys.path.append(util_path)
sys.path.append(train_path)
sys.path.append(prom_path)

from prom.query import PrometheusClient

#SAMPLING_INTERVAL = PROM_QUERY_INTERVAL
#SAMPLING_INTERVAL = getConfig('SAMPLING_INTERVAL', SAMPLING_INTERVAL)
#SAMPLING_INTERVAL = int(SAMPLING_INTERVAL)
SAMPLING_INTERVAL = 20


from train.pipelines.XGBoostRegressionStandalonePipeline.pipe import XGBoostRegressionStandalonePipeline
from train.pipe_util import XGBoostRegressionTrainType


if __name__ == '__main__':
    prom_client = PrometheusClient()
    while True:
       xgb_pipeline = XGBoostRegressionStandalonePipeline(XGBoostRegressionTrainType.KFoldCrossValidation, "XGBoost/")
       xgb_pipeline.train(prom_client)
        
       time.sleep(SAMPLING_INTERVAL)
    # xgb_pipeline = XGBoostRegressionStandalonePipeline(XGBoostRegressionTrainType.TrainTestSplitFit, "XGBoost/")
    # results, model_desc = xgb_pipeline.predict([[18374, 37371, 37372], [483838, 28383, 23774]])
    # print(results)
    # print(model_desc)