# prom_test.py
#   call prometheus module to query data save to query_data/<export_metric>.csv

import os
import sys

server_path = os.path.join(os.path.dirname(__file__), '../src')
util_path = os.path.join(os.path.dirname(__file__), '../src/util')
train_path = os.path.join(os.path.dirname(__file__), '../src/train')
prom_path = os.path.join(os.path.dirname(__file__), '../src/train/prom')

sys.path.append(server_path)
sys.path.append(util_path)
sys.path.append(train_path)
sys.path.append(prom_path)

prom_output_path = os.path.join(os.path.dirname(__file__), 'data', 'prom_output')

from prom.query import PrometheusClient

if __name__ == '__main__':
    prom_client = PrometheusClient()
    prom_client.query()
    results = prom_client.snapshot_query_result()
    # save query data in csv
    for metric, data in results.items():
        print(metric)
        print(data.head())