import os
import sys

src_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(src_path)

from prometheus_api_client import PrometheusConnect
import datetime

import pandas as pd
from util.config import getConfig

PROM_SERVER = 'http://localhost:9090'
PROM_SSL_DISABLE = 'True'
PROM_HEADERS = ''
PROM_QUERY_INTERVAL = 300
PROM_QUERY_STEP = 3

PROM_SERVER = getConfig('PROM_SERVER', PROM_SERVER)
PROM_HEADERS = getConfig('PROM_HEADERS', PROM_HEADERS)
PROM_HEADERS = None if PROM_HEADERS == '' else PROM_HEADERS
PROM_SSL_DISABLE = True if getConfig('PROM_SSL_DISABLE', PROM_SSL_DISABLE).lower() == 'true' else False
PROM_QUERY_INTERVAL = getConfig('PROM_QUERY_INTERVAL', PROM_QUERY_INTERVAL)

metric_prefix = "kepler_"
TIMESTAMP_COL = "timestamp"
PACKAGE_COL = "package"
SOURCE_COL = "source"
MODE_COL = "mode"

def get_energy_unit(component):
    if component in ["package", "core", "uncore", "dram"]:
        return "package"
    return None

def generate_dataframe_from_response(query_metric, prom_response):
    items = []
    for res in prom_response:
        metric_item = res['metric']
        for val in res['values']:
            # labels
            item = metric_item.copy()
            # timestamp
            item[TIMESTAMP_COL] = val[0]
            # value
            item[query_metric] = val[1] 
            items += [item]
    df = pd.DataFrame(items)
    return df

class PrometheusClient():
    def __init__(self):
        self.prom = PrometheusConnect(url=PROM_SERVER, headers=PROM_HEADERS, disable_ssl=PROM_SSL_DISABLE)
        self.interval = int(PROM_QUERY_INTERVAL)
        self.step = int(PROM_QUERY_STEP)
        self.latest_query_result = dict()

    def query(self):
        available_metrics = self.prom.all_metrics()
        queries = [m for m in available_metrics if metric_prefix in m]
        end = datetime.datetime.now()
        start = end - datetime.timedelta(seconds=self.interval)
        self.latest_query_result = dict()
        print(self.interval, self.step, start, end)
        for query_metric in queries:
            prom_response = self.prom.custom_query_range(query_metric, start, end, self.step, None)
            self.latest_query_result[query_metric] = generate_dataframe_from_response(query_metric, prom_response)

    def snapshot_query_result(self):
        return {metric: data for metric, data in self.latest_query_result.items() if len(data) > 0}