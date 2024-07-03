import os
import sys

server_path = os.path.join(os.path.dirname(__file__), "../server")
util_path = os.path.join(os.path.dirname(__file__), "../server/util")
prom_path = os.path.join(os.path.dirname(__file__), "../server/prom")

sys.path.append(server_path)
sys.path.append(util_path)
sys.path.append(prom_path)

from prom.prom_query import NODE_STAT_QUERY, POD_STAT_QUERY, PROM_QUERY_INTERVAL, PrometheusClient
from train.train_types import FeatureGroups
from util.config import getConfig

SAMPLING_INTERVAL = PROM_QUERY_INTERVAL
SAMPLING_INTERVAL = getConfig("SAMPLING_INTERVAL", SAMPLING_INTERVAL)
SAMPLING_INTERVAL = int(SAMPLING_INTERVAL)

if __name__ == "__main__":
    prom_client = PrometheusClient()
    prom_client.query()
    results = prom_client.latest_query_result
    prom_output_path = os.path.join(os.path.dirname(__file__), "query_data")
    # save query data in csv
    for query, result in results.items():
        result.to_csv(f"{prom_output_path}/{query}.csv")
    # print data get by feature list
    for query_metric in [POD_STAT_QUERY, NODE_STAT_QUERY]:
        for fg, features in FeatureGroups.items():
            data = prom_client.get_data(query_metric, features)
            print(f"Query: {query_metric} Type: {fg.name} Features: {features}")
            print(None if data is None else data.head())
