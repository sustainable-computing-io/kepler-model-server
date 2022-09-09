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
from train.keras_pipe import KerasFullPipeline
from util.config import getConfig

SAMPLING_INTERVAL = PROM_QUERY_INTERVAL
SAMPLING_INTERVAL = getConfig('SAMPLING_INTERVAL', SAMPLING_INTERVAL)
SAMPLING_INTERVAL = int(SAMPLING_INTERVAL)

pipelines = [KerasFullPipeline()]

if __name__ == '__main__':
    prom_client = PrometheusClient()
    while True:
        prom_client.query()
        for pipeline in pipelines:
            pipeline.train(prom_client)
        time.sleep(SAMPLING_INTERVAL)