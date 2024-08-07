import os

# WARN: check if this test is still needed

from kepler_model.train.prom.prom_query import PrometheusClient, PROM_QUERY_INTERVAL, POD_STAT_QUERY, NODE_STAT_QUERY
from kepler_model.util.config import getConfig

from kepler_model.util.train_types import FeatureGroups


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
        result.to_csv("{}/{}.csv".format(prom_output_path, query))
    # print data get by feature list
    for query_metric in [POD_STAT_QUERY, NODE_STAT_QUERY]:
        for fg, features in FeatureGroups.items():
            data = prom_client.get_data(query_metric, features)
            print("Query: {} Type: {} Features: {}".format(query_metric, fg.name, features))
            print(None if data is None else data.head())

