worklaod = "coremark_4threads_to_32threads_5rep_ondemand_demo"
input_data_folder = "query_data/" + worklaod
output_data_folder = "predict_data"
query = "node_energy_stat"
pipeline_name = 'KerasCompFullPipeline'

import os
import sys

train_path = os.path.join(os.path.dirname(__file__), '../server/train')
sys.path.append(train_path)
import importlib
pipeline_module = importlib.import_module('pipelines.{}.pipe'.format(pipeline_name))
pipeline = getattr(pipeline_module, pipeline_name)()

if not os.path.exists(output_data_folder):
    os.mkdir(output_data_folder)

import json
import pandas as pd
from prometheus_client.core import REGISTRY, GaugeMetricFamily
from prometheus_client import start_wsgi_server
import random

class RandomNumberCollector(object):
    def __init__(self):
        pass
    def collect(self):
        gauge = GaugeMetricFamily("random_number", "A random number generator", labels=["randomNum"])
        gauge.add_metric(['random_num'], random.randint(1, 20))
        yield gauge

def apply_predict():
    prom_output_path = os.path.join(os.path.dirname(__file__), input_data_folder)
    csv_filepath = "{}/{}.csv".format(prom_output_path, query)
    output_filepath = "{}/{}.json".format(output_data_folder, worklaod)
    print(csv_filepath)
    if os.path.exists(csv_filepath):
        data = pd.read_csv(csv_filepath)
        results = pipeline.predict(data)
        detail = dict()
        for comp, result in results.items():
            detail[comp] = result.squeeze().tolist()
        with open(output_filepath, "w") as f:
            json.dump(detail, f)

import time
if __name__ == "__main__":
    start_wsgi_server(9000)
    REGISTRY.register(RandomNumberCollector())
    apply_predict()
    while True: 
        time.sleep(3)