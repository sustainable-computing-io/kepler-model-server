import os
import sys
import time

util_path = os.path.join(os.path.dirname(__file__), 'util')
train_path = os.path.join(os.path.dirname(__file__), 'train')
prom_path = os.path.join(os.path.dirname(__file__), 'prom')
sys.path.append(util_path)
sys.path.append(train_path)
sys.path.append(prom_path)

from prom.query import PrometheusClient, PROM_QUERY_INTERVAL
from util.config import getConfig

SAMPLING_INTERVAL = PROM_QUERY_INTERVAL
SAMPLING_INTERVAL = getConfig('SAMPLING_INTERVAL', SAMPLING_INTERVAL)
SAMPLING_INTERVAL = int(SAMPLING_INTERVAL)


from train.pipelines.XGBoostRegressionStandalonePipeline.pipe import XGBoostRegressionStandalonePipeline
from train.pipe_util import XGBoostRegressionTrainType
import pandas as pd
def read_sample_query_results():
   prom_output_path = prom_output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests', 'data', 'prom_output')

   results = dict()
   metric_filenames = [ metric_filename for metric_filename in os.listdir(prom_output_path) ]
   for metric_filename in metric_filenames:
      metric = metric_filename.replace(".csv", "")
      filepath = os.path.join(prom_output_path, metric_filename)
      results[metric] = pd.read_csv(filepath)
   return results

if __name__ == '__main__':
   prom_client = PrometheusClient()
   while True:
      #xgb_container_pipeline = XGBoostRegressionStandalonePipeline(XGBoostRegressionTrainType.KFoldCrossValidation, "XGBoost/", node_level=False)
      xgb_node_pipeline = XGBoostRegressionStandalonePipeline(XGBoostRegressionTrainType.KFoldCrossValidation, "XGBoost/", node_level=True)
      #xgb_container_pipeline.train(prom_client)
      xgb_node_pipeline.train(prom_client)
        
      time.sleep(SAMPLING_INTERVAL)
