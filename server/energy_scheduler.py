import schedule
from prometheus_datapipeline_functions import retrieve_and_clean_prometheus_energy_metrics, create_prometheus_core_dataset, create_prometheus_dram_dataset
from kepler_model_trainer import train_model_given_data_and_type
import time

#TODO: Test with Kepler on AWS

# Temporary scheduler to scrape prometheus metrics once every day
def energy_prometheus_pipeline():
    energy_data_dict = retrieve_and_clean_prometheus_energy_metrics()
    core_train, core_val, core_test = create_prometheus_core_dataset(energy_data_dict['cpu_architecture'], energy_data_dict['curr_cpu_cycles'], energy_data_dict['current_cpu_instructions'], energy_data_dict['curr_cpu_time'], energy_data_dict['curr_energy_in_core'] )
    dram_train, dram_val, dram_test = create_prometheus_dram_dataset(energy_data_dict['cpu_architecture'], energy_data_dict['curr_cache_misses'], energy_data_dict['curr_resident_memory'], energy_data_dict['curr_energy_in_dram'])
    train_model_given_data_and_type(core_train, core_val, core_test, 'core_model')
    train_model_given_data_and_type(dram_train, dram_val, dram_test, 'dram_model')


schedule.every().day.at("00:00").do(energy_prometheus_pipeline)

while True:
    schedule.run_pending()
    time.sleep(1)