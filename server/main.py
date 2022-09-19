import os
import sys
import time

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

from prometheus_datapipeline_functions import retrieve_and_clean_prometheus_energy_metrics, create_prometheus_core_dataset, create_prometheus_dram_dataset
from train.pipelines.KerasFullPipeline.pipe_util import train_model_given_data_and_type

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