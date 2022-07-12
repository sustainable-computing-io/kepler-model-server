from numpy import test
from prometheus_datapipeline_functions import retrieve_and_clean_prometheus_energy_metrics, create_prometheus_core_dataset, create_prometheus_dram_dataset, retrieve_dummy_prometheus_metrics
from kepler_model_trainer import train_model_given_data_and_type
import time
import os

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

if __name__ == "__main__":
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
   