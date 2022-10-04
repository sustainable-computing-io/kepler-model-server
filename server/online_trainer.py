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

pipeline_names = ['KerasFullPipeline']
grouped_pipelines = dict()

for pipeline_name in pipeline_names:
    pipeline_path = os.path.join(os.path.dirname(__file__), 'train/pipelines/{}'.format(pipeline_name))
    sys.path.append(pipeline_path)
    import importlib
    pipeline_module = importlib.import_module('train.pipelines.{}.pipe'.format(pipeline_name))
    pipeline = getattr(pipeline_module, pipeline_name)()
    output_type = pipeline.output_type.name
    if output_type not in grouped_pipelines:
        grouped_pipelines[output_type] = []
    grouped_pipelines[output_type] += [pipeline]

def run_train(pipeline, prom_client):
    pipeline.train(prom_client)

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

if __name__ == '__main__':
    prom_client = PrometheusClient()
    while True:
        prom_client.query()

        # start the thread pool
        with ThreadPoolExecutor(2) as executor:
            futures = []
            for output_type, pipelines in grouped_pipelines.items():
                for pipeline in pipelines:
                    future = executor.submit(run_train, pipeline, prom_client)
                    futures += [future]
            print('Waiting for tasks to complete...')
            wait(futures)
            print('All trained!')
        time.sleep(SAMPLING_INTERVAL)