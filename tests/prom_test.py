# prom_test.py
#   - prom_client.query 
#   - prom_client.snapshot_query_result
#
# save response to prom_output_path/prom_output_filename.json
#
# To use output:
# from prom_test import get_prom_output
# response = get_prom_response()
# or
# query_result = get_query_results()


# import external src
import os
import sys

#################################################################
# import internal src 
src_path = os.path.join(os.path.dirname(__file__), '../src')
sys.path.append(src_path)
#################################################################

from train.prom import PrometheusClient, prom_responses_to_results
from util import save_json, load_json

prom_output_path = os.path.join(os.path.dirname(__file__), 'data', 'prom_output')
prom_output_filename = "prom_response"

def get_prom_response(save_path=prom_output_path, save_name=prom_output_filename):
    return load_json(save_path, save_name)

def get_query_results(save_path=prom_output_path, save_name=prom_output_filename):
    response = get_prom_response(save_path=save_path, save_name=save_name)
    return prom_responses_to_results(response)

def process(save_path=prom_output_path, save_name=prom_output_filename, server=None, interval=None, step=None):
    if server is not None:
        os.environ['PROM_SERVER'] = server
    if interval is not None:
        os.environ['PROM_QUERY_INTERVAL'] = interval
    if step is not None:
        os.environ['PROM_QUERY_STEP'] = step
    prom_client = PrometheusClient()
    response_dict = prom_client.query()
    results = prom_client.snapshot_query_result()
    print("Available metrics: ", results.keys())
    # print query data in csv
    for metric, data in results.items():
        print(metric)
        print(data.head())
    save_json(save_path, save_name, response_dict)

if __name__ == '__main__':
    process()
