import os
import sys

server_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(server_path)

from prometheus_api_client import PrometheusConnect
import datetime

import pandas as pd
from util.config import getConfig

PROM_SERVER = 'http://localhost:9090'
PROM_SSL_DISABLE = 'True'
PROM_HEADERS = ''
PROM_QUERY_INTERVAL = 20
PROM_QUERY_STEP = 3

PROM_SERVER = getConfig('PROM_SERVER', PROM_SERVER)
PROM_HEADERS = getConfig('PROM_HEADERS', PROM_HEADERS)
PROM_HEADERS = None if PROM_HEADERS == '' else PROM_HEADERS
PROM_SSL_DISABLE = True if getConfig('PROM_SSL_DISABLE', PROM_SSL_DISABLE).lower() == 'true' else False
PROM_QUERY_INTERVAL = getConfig('PROM_QUERY_INTERVAL', PROM_QUERY_INTERVAL)

NODE_STAT_QUERY = 'node_energy_stat'
POD_STAT_QUERY = 'pod_energy_stat'
PKG_ENERGY_QUERY = 'node_package_energy_millijoule'
POD_USAGE_PER_CPU_QUERY = 'pod_cpu_cpu_time_us'
CPU_FREQUENCY_QUERY = 'node_cpu_scaling_frequency_hertz'

NODE_STAT_QUERY = getConfig('NODE_STAT_QUERY', NODE_STAT_QUERY)
POD_STAT_QUERY = getConfig('POD_STAT_QUERY', POD_STAT_QUERY)
PKG_ENERGY_QUERY = getConfig('PKG_ENERGY_QUERY', PKG_ENERGY_QUERY)
POD_USAGE_PER_CPU_QUERY = getConfig('POD_USAGE_PER_CPU_QUERY', POD_USAGE_PER_CPU_QUERY)
CPU_FREQUENCY_QUERY = getConfig('CPU_FREQUENCY_QUERY', CPU_FREQUENCY_QUERY)

QUERIES = [NODE_STAT_QUERY, POD_STAT_QUERY, PKG_ENERGY_QUERY, POD_USAGE_PER_CPU_QUERY, CPU_FREQUENCY_QUERY]

def transform_float(val):
    try: 
        val = float(val)
    except:
        pass
    return val

class PrometheusClient():
    def __init__(self):
        self.prom = PrometheusConnect(url=PROM_SERVER, headers=PROM_HEADERS, disable_ssl=PROM_SSL_DISABLE)
        self.interval = int(PROM_QUERY_INTERVAL)
        self.step = int(PROM_QUERY_STEP)
        self.latest_query_result = dict()

    def query(self):
        available_metrics = self.prom.all_metrics()
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(seconds=self.interval)
        self.latest_query_result = dict()
        for query_metric in QUERIES:
            if query_metric not in available_metrics:
                self.latest_query_result[query_metric] = pd.DataFrame()
                print("No {} exported".format(query_metric))
                continue
            prom_response = self.prom.custom_query_range(query_metric, start, end, self.step, None)
            items = []
            for res in prom_response:
                metric_item = res['metric']
                for val in res['values']:
                    item = metric_item.copy()
                    item['timestamp'] = val[0]
                    item['value'] = val[1] 
                    items += [item]
            df = pd.DataFrame(items) 
            df.columns = df.columns.str.replace("curr_", "")
            df.columns = df.columns.str.replace("node_", "")
            if len(df) > 0:
                df[query_metric] = df['value']
                for col in df.columns:
                    df[col] = df[col].transform(transform_float)
                df.drop(columns=['value'], inplace=True)
            self.latest_query_result[query_metric] = df
        
    def get_data(self, query_metric, features):
        if len(self.latest_query_result[query_metric]) == 0:
            return None
        if features is None:
            # get all columns
            return self.latest_query_result[query_metric]
        try: 
            data = self.latest_query_result[query_metric][features + [query_metric]]
        except:
            data = None
        return data

