import os 
import sys

server_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(server_path)

from train.train_types import FeatureGroups
from query import PrometheusClient, POD_STAT_QUERY, NODE_STAT_QUERY

if __name__ == "__main__":
    prom_client = PrometheusClient()
    prom_client.query()
    for query_metric in [POD_STAT_QUERY, NODE_STAT_QUERY]: 
        for fg, features in FeatureGroups.items():
            data = prom_client.get_data(query_metric, features)
            print('Query: {} Type: {} Features: {}'.format(query_metric, fg.name, features))
            print(data if data is not None else None)