import os
import sys

import datetime

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

from prom_types import TIMESTAMP_COL, PROM_SERVER, PROM_HEADERS, PROM_SSL_DISABLE, PROM_QUERY_INTERVAL, PROM_QUERY_STEP, metric_prefix

from prometheus_api_client import PrometheusConnect

import pandas as pd

def _range_queries(prom, metric_list, start, end, step, params=None):
    response = dict()
    for metric in metric_list:
        response[metric] = prom.custom_query_range(metric, start, end, step, params)
    return response

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
            item[query_metric] = float(val[1]) 
            items += [item]
    df = pd.DataFrame(items)
    return df

def prom_responses_to_results(prom_responses):
    results = dict()
    for query_metric, prom_response in prom_responses.items():
        results[query_metric] = generate_dataframe_from_response(query_metric, prom_response)
    return results

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
        response_dict = _range_queries(self.prom, queries, start, end, self.step, None)
        for query_metric, prom_response in response_dict.items():
            self.latest_query_result[query_metric] = generate_dataframe_from_response(query_metric, prom_response)
        return response_dict
        
    def snapshot_query_result(self):
        return {metric: data for metric, data in self.latest_query_result.items() if len(data) > 0}