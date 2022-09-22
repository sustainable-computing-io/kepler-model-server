import os
import sys
import time
import requests
import numpy as np

util_path = os.path.join(os.path.dirname(__file__), 'util')
train_path = os.path.join(os.path.dirname(__file__), 'train')
prom_path = os.path.join(os.path.dirname(__file__), 'prom')

sys.path.append(util_path)
sys.path.append(train_path)
sys.path.append(prom_path)

from prom.query import PrometheusClient, PROM_QUERY_INTERVAL
from util.config import getConfig

SAMPLING_INTERVAL = PROM_QUERY_INTERVAL
SAMPLING_INTERVAL = getConfig('SAMPLING_INTERVAL', SAMPLING_INTERVAL)
SAMPLING_INTERVAL = int(SAMPLING_INTERVAL)

pipeline_names = ['KerasFullPipeline']
grouped_pipelines = dict()

for pipeline_name in pipeline_names:
    pipeline_path = os.path.join(os.path.dirname(__file__), 'train/pipelines/{}'.format(pipeline_name))
    sys.path.append(pipeline_path)
    import importlib
    pipeline_module = importlib.import_module('train.pipelines.{}.pipe'.format(pipeline_name))
    pipeline = getattr(pipeline_module, pipeline_name)()
    output_type = pipeline.output_type.name
    if output_type not in grouped_pipelines:
        grouped_pipelines[output_type] = []
    grouped_pipelines[output_type] += [pipeline]

def run_train(pipeline, prom_client):
    pipeline.train(prom_client)

from train.pipelines.KerasFullPipeline.pipe import create_prometheus_core_dataset, create_prometheus_dram_dataset
from train.pipelines.KerasFullPipeline.pipe_util import train_model_given_data_and_type


def query_prometheus(query, interval, endpoint):
    query_str= 'query={}[{}]'.format(query, interval)
    print('query {}'.format(query_str))
    return requests.get(url = endpoint, params = query_str).json()


#Testing using a Prometheus exporter which monitors itself for metrics
def retrieve_dummy_prometheus_metrics(query, interval, endpoint):
    return query_prometheus(query, interval, endpoint)


def retrieve_and_clean_prometheus_energy_metrics(query, interval, endpoint):
    energy_metrics_dataset = query_prometheus(query, interval, endpoint)
    # Retrieve all the desired energy related features in the form of tensors
    curr_cpu_cycles = []
    curr_cpu_instructions = []
    curr_cpu_time = []
    curr_energy_in_core = []
    cpu_architecture = []
    curr_resident_memory = []
    curr_cache_misses = []
    curr_energy_in_dram = []
    result = energy_metrics_dataset['data']['result']
    for x in range(len(result)):
        raw_metrics = result[x]
        refined_metrics = raw_metrics['metric']
        curr_cpu_cycles_val = float(refined_metrics['curr_cpu_cycles'])
        curr_cpu_instructions_val = float(refined_metrics['curr_cpu_instructions'])
        curr_cpu_time_val = float(refined_metrics['curr_cpu_time'])
        cpu_architecture_val = refined_metrics['cpu_architecture']
        curr_energy_in_core_val = float(refined_metrics['curr_energy_in_core'])
        curr_energy_in_dram_val = float(refined_metrics['curr_energy_in_dram'])
        curr_cache_misses_val = float(refined_metrics['curr_cache_misses'])
        curr_resident_memory_val = float(refined_metrics['curr_resident_memory'])
        print("{}:{}:{}:{}:{}:{}:{}:{}".format(curr_cpu_cycles_val, curr_cpu_instructions_val, curr_cpu_time_val, cpu_architecture_val, curr_energy_in_core_val, curr_energy_in_dram_val, curr_cache_misses_val, curr_resident_memory_val))
        curr_cpu_cycles.append(curr_cpu_cycles_val)
        curr_cpu_instructions.append(curr_cpu_instructions_val)
        curr_cpu_time.append(curr_cpu_time_val)
        cpu_architecture.append(cpu_architecture_val)
        curr_energy_in_core.append(curr_energy_in_core_val)
        curr_energy_in_dram.append(curr_energy_in_dram_val)
        curr_cache_misses.append(curr_cache_misses_val)
        curr_resident_memory.append(curr_resident_memory_val)

    curr_cpu_cycles = np.hstack(curr_cpu_cycles)
    curr_cpu_instructions = np.hstack(curr_cpu_instructions)
    curr_cpu_time = np.hstack(curr_cpu_time)
    cpu_architecture = np.hstack(cpu_architecture)
    curr_energy_in_core = np.hstack(curr_energy_in_core)
    energy_data_dict = {'curr_cpu_cycles': curr_cpu_cycles, 'current_cpu_instructions': curr_cpu_instructions,
                        'curr_cpu_time': curr_cpu_time, 'cpu_architecture': cpu_architecture, 'curr_energy_in_core': curr_energy_in_core,
                        'curr_energy_in_dram': curr_energy_in_dram, 'curr_cache_misses': curr_cache_misses, 'curr_resident_memory': 
                        curr_resident_memory}
    return energy_data_dict

# scrape prometheus energy metrics and train models
def energy_prometheus_pipeline(query, interval, endpoint):
    energy_data_dict = retrieve_and_clean_prometheus_energy_metrics(query, interval, endpoint)
    core_train, core_val, core_test = create_prometheus_core_dataset(energy_data_dict['cpu_architecture'], energy_data_dict['curr_cpu_cycles'], energy_data_dict['current_cpu_instructions'], energy_data_dict['curr_cpu_time'], energy_data_dict['curr_energy_in_core'] )
    dram_train, dram_val, dram_test = create_prometheus_dram_dataset(energy_data_dict['cpu_architecture'], energy_data_dict['curr_cache_misses'], energy_data_dict['curr_resident_memory'], energy_data_dict['curr_energy_in_dram'])
    train_model_given_data_and_type(core_train, core_val, core_test, 'core_model')
    train_model_given_data_and_type(dram_train, dram_val, dram_test, 'dram_model')

# scrape prometheus and dummy test metrics
def dummy_prometheus_pipeline(query, interval, endpoint):
    test_dataset = retrieve_and_clean_prometheus_energy_metrics(query, interval, endpoint)
    print(test_dataset)


def run_energy_scheduler_pipeline():
    #to run on schedule:
    #import schedule
    #schedule.every().day.at("00:00").do(energy_prometheus_pipeline)
    #schedule.every(10).seconds.do(dummy_prometheus_pipeline)
    endpoint = os.getenv('PROMETHEUS_ENDPOINT', default='http://localhost:9090/api/v1/query')
    query = os.getenv('PROMETHEUS_QUERY', default='node_energy_stat')
    interval_sec = os.getenv('PROMETHEUS_QUERY_INTERVAL', default=20)
    interval = '{}s'.format(interval_sec)
    # To test:
    # dummy_prometheus_pipeline(query, interval, endpoint) 
    start = time.time()

    energy_prometheus_pipeline(query, interval, endpoint)
    end = time.time()
    while True:
        time_lapsed = end - start
        if time_lapsed > interval_sec:
            start = time.time()
            energy_prometheus_pipeline(query, '{}s'.format(int(time_lapsed)), endpoint)
        else: 
            time.sleep(int(interval_sec - time_lapsed))
        end = time.time()


from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait


if __name__ == '__main__':
    prom_client = PrometheusClient()
    while True:
        prom_client.query()

        # start the thread pool
        with ThreadPoolExecutor(2) as executor:
            futures = []
            for output_type, pipelines in grouped_pipelines.items():
                for pipeline in pipelines:
                    future = executor.submit(run_train, pipeline, prom_client)
                    futures += [future]
            print('Waiting for tasks to complete...')
            wait(futures)
            print('All trained!')
        time.sleep(SAMPLING_INTERVAL)