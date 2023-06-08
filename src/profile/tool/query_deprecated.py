from prometheus_api_client import PrometheusConnect
import datetime
import json
import os

query_result_folder = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'resource', 'query_response')

metric_prefix = "kepler_"

step = "3"
params = None

UTC_OFFSET_TIMEDELTA = datetime.datetime.utcnow() - datetime.datetime.now()

def _range_queries(prom, metric_list, start, end):
    global metrics, step, params 
    response = dict()
    for metric in metric_list:
        response[metric] = prom.custom_query_range(metric, start, end, step, params)
    return response


def query(prom, metric_list, start, end, offset=0):
    print("collecting from {0} to {1}".format(start, end))
    try:
        response = _range_queries(prom, metric_list, start - UTC_OFFSET_TIMEDELTA, end - UTC_OFFSET_TIMEDELTA)
    except:
        print("failed to collect {}".format(prom))
        return None
    return response

def save(file_name, response):
    output_flename = "{0}/response.json".format("/".join(file_name.split("/")[0:-1]))
    with open(output_flename, "w") as outfile:
        json.dump(response, outfile)

import sys
import yaml
from yaml.loader import SafeLoader

def extract_time(file_name):
    with open(file_name, "r") as f:
        data = yaml.load(f, Loader=SafeLoader)
        start_str = data["metadata"]["creationTimestamp"]
        start = datetime.datetime.strptime(start_str, '%Y-%m-%dT%H:%M:%SZ')
        end_str = data["status"]["results"][-1]["repetitions"][-1]["pushedTime"].split(".")[0]
        end = datetime.datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
    return start, end

def convert_time(timestr):
    datetime_object = datetime.datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S') # in UTC
    return datetime_object - UTC_OFFSET_TIMEDELTA

if __name__ == "__main__":
    prom = PrometheusConnect(url ="http://127.0.0.1:30090", disable_ssl=True)
    metric_list = [m for m in prom.all_metrics()if metric_prefix in m]
    print("metrics:", metric_list)

    if len(sys.argv) < 2:
        print("enter cr file")
        exit()

    file_name = sys.argv[1]
    start, end = extract_time(file_name)

    response = query(prom, metric_list, start, end)
    save(file_name, response)