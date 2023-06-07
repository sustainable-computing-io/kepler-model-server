# deprecated
import os
import sys

os.environ['MODEL_PATH'] = "../tests/test_models"

server_path = os.path.join(os.path.dirname(__file__), '../server')
util_path = os.path.join(os.path.dirname(__file__), '../server/util')
train_path = os.path.join(os.path.dirname(__file__), '../server/train')
prom_path = os.path.join(os.path.dirname(__file__), '../server/prom')

sys.path.append(server_path)
sys.path.append(util_path)
sys.path.append(train_path)
sys.path.append(prom_path)

from prom.prom_query import PrometheusClient, PROM_QUERY_INTERVAL, QUERIES
from util.config import getConfig

import pandas as pd

SAMPLING_INTERVAL = PROM_QUERY_INTERVAL
SAMPLING_INTERVAL = getConfig('SAMPLING_INTERVAL', SAMPLING_INTERVAL)
SAMPLING_INTERVAL = int(SAMPLING_INTERVAL)

pipeline_names = ['KerasFullPipeline', 'KerasCompWeightFullPipeline', 'KerasCompFullPipeline']
grouped_pipelines = dict()

for pipeline_name in pipeline_names:
    pipeline_path = os.path.join(os.path.dirname(__file__), '../server/train/pipelines/{}'.format(pipeline_name))
    sys.path.append(pipeline_path)
    import importlib
    pipeline_module = importlib.import_module('train.pipelines.{}.pipe'.format(pipeline_name))
    pipeline = getattr(pipeline_module, pipeline_name)()
    output_type = pipeline.output_type.name
    if output_type not in grouped_pipelines:
        grouped_pipelines[output_type] = []
    grouped_pipelines[output_type] += [pipeline]

def load_query_data(prom_client):
    prom_output_path = os.path.join(os.path.dirname(__file__), 'query_data')
    # save query data in csv
    for query in QUERIES:
        csv_filepath = "{}/{}.csv".format(prom_output_path, query)
        if os.path.exists(csv_filepath):
            data = pd.read_csv(csv_filepath)
            if len(data) > 0:
                prom_client.latest_query_result[query] = data


def run_train(pipeline, prom_client):
    pipeline.train(prom_client)
    return "{} Done".format(pipeline.model_name)

def execute():
    with ThreadPoolExecutor(2) as executor:
        futures = []
        for output_type, pipelines in grouped_pipelines.items():
            for pipeline in pipelines:
                future = executor.submit(run_train, pipeline, prom_client)
                futures += [future]
        print('Waiting for {} tasks to complete...'.format(len(futures)))
        for ret in as_completed(futures):
            print(ret.result())
        print('All trained!')

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

if __name__ == '__main__':
    prom_client = PrometheusClient()
    load_query_data(prom_client)
    # initial train
    execute()
    # online train
    execute()
