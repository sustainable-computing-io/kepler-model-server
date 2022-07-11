from hashlib import new
import requests
import numpy as np
import tensorflow as tf

#TODO: Test with Kepler on AWS

# Returns the list for each metric
def scrape_prometheus_metrics(length):
    #requests.get("http://10.43.221.48:9090/api/v1/query?query=node_energy_stat[{}]".format(length))
    pass

#Testing using a Prometheus exporter which monitors itself for metrics
def retrieve_dummy_prometheus_metrics():
    return scrape_prometheus_metrics("go_memstats_alloc_bytes_total", 5, "http://localhost:9091")

def retrieve_and_clean_prometheus_energy_metrics():
    energy_metrics = scrape_prometheus_metrics(100) # expects result list
    # Retrieve all the desired energy related features in the form of tensors
    curr_cpu_cycles = []
    curr_cpu_instructions = []
    curr_cpu_time = []
    curr_energy_in_core = []
    cpu_architecture = []
    curr_resident_memory = []
    curr_cache_misses = []
    curr_energy_in_dram = []
    for energy_metric in energy_metrics:
        refined_metrics = energy_metric['metric']

        curr_cpu_cycles.append(refined_metrics['curr_cpu_cycles'])
        curr_cpu_instructions.append(refined_metrics['current_cpu_instructions'])
        curr_cpu_time.append(refined_metrics['curr_cpu_time'])
        cpu_architecture.append(refined_metrics['cpu_architecture'])
        curr_energy_in_core.append(refined_metrics['curr_energy_in_core'])
        curr_energy_in_dram.append(refined_metrics['curr_energy_in_dram'])
        curr_cache_misses.append(refined_metrics['curr_cache_misses'])
        curr_resident_memory.append(refined_metrics['curr_resident_memory'])

    curr_cpu_cycles = np.concatenate(curr_cpu_cycles)
    curr_cpu_instructions = np.concatenate(curr_cpu_instructions)
    curr_cpu_time = np.concatenate(curr_cpu_time)
    cpu_architecture = np.concatenate(cpu_architecture)
    curr_energy_in_core = np.concatenate(curr_energy_in_core)
    energy_data_dict = {'curr_cpu_cycles': curr_cpu_cycles, 'current_cpu_instructions': curr_cpu_instructions,
                        'curr_cpu_time': curr_cpu_time, 'cpu_architecture': cpu_architecture, 'curr_energy_in_core': curr_energy_in_core,
                        'curr_energy_in_dram': curr_energy_in_dram, 'curr_cache_misses': curr_cache_misses, 'curr_resident_memory': 
                        curr_resident_memory}
    return energy_data_dict


def create_prometheus_core_dataset(cpu_architecture, curr_cpu_cycles, curr_cpu_instructions, curr_cpu_time, curr_energy_in_core): #query: str, length: int, endpoint: str

    # Create Desired Datasets for Core Model
    features_dict_core = {'cpu_architecture': cpu_architecture, 'curr_cpu_cycles': curr_cpu_cycles, 'current_cpu_instructions': curr_cpu_instructions, 'curr_cpu_time': curr_cpu_time}
    refined_dataset_core = tf.data.Dataset.from_tensor_slices((features_dict_core, curr_energy_in_core))

    refined_dataset_core_size = refined_dataset_core.cardinality().numpy()

    assert(refined_dataset_core_size >= 5)
    train_size = int(refined_dataset_core_size*0.6)
    val_size = int(refined_dataset_core_size*0.2)
    train_dataset_core = refined_dataset_core.take(train_size)
    val_dataset_core = refined_dataset_core.skip(train_size).take(val_size)
    test_dataset_core = refined_dataset_core.skip(train_size).skip(val_size)
    
    train_dataset_core = train_dataset_core.shuffle(buffer_size=train_size).repeat(4).batch(32)
    val_dataset_core = val_dataset_core.batch(32)
    test_dataset_core = test_dataset_core.batch(32)

    return train_dataset_core, val_dataset_core, test_dataset_core


def create_prometheus_dram_dataset(cpu_architecture, curr_cache_misses, curr_resident_memory, curr_energy_in_dram):
    # Create Desired Dataset for Dram Model
    features_dict_dram = {'cpu_architecture': cpu_architecture, 'curr_cache_misses': curr_cache_misses, 'curr_resident_memory': 
                        curr_resident_memory}
    refined_dataset_dram = tf.data.Dataset.from_tensor_slices((features_dict_dram, curr_energy_in_dram))
    refined_dataset_dram_size = refined_dataset_dram.cardinality().numpy()
    assert(refined_dataset_dram_size >= 5)
    train_size = int(refined_dataset_dram_size*0.6)
    val_size = int(refined_dataset_dram_size*0.2)
    train_dataset_dram = refined_dataset_dram.take(train_size)
    val_dataset_dram = refined_dataset_dram.skip(train_size).take(val_size)
    test_dataset_dram = refined_dataset_dram.skip(train_size).skip(val_size)
    
    train_dataset_dram = train_dataset_dram.shuffle(buffer_size=train_size).repeat(4).batch(32)
    val_dataset_dram = val_dataset_dram.batch(32)
    test_dataset_dram = test_dataset_dram.batch(32)

    return train_dataset_dram, val_dataset_dram, test_dataset_dram

