import datetime

from prometheus_api_client import PrometheusConnect

from kepler_model.util.prom_types import (
    PROM_QUERY_INTERVAL,
    PROM_QUERY_STEP,
    PROM_SERVER,
    PROM_SSL_DISABLE,
    generate_dataframe_from_response,
    metric_prefix,
)

UTC_OFFSET_TIMEDELTA = datetime.datetime.utcnow() - datetime.datetime.now()


def _range_queries(prom, metric_list, start, end, step, params=None):
    response = dict()
    for metric in metric_list:
        response[metric] = prom.custom_query_range(metric, start, end, step, params)
    return response


class PrometheusClient:
    def __init__(self):
        self.prom = PrometheusConnect(url=PROM_SERVER, disable_ssl=PROM_SSL_DISABLE)
        self.interval = PROM_QUERY_INTERVAL
        self.step = PROM_QUERY_STEP
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
