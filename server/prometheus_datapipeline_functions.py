from hashlib import new
import requests
import numpy as np
import tensorflow as tf
import requests

def query_prometheus(query, interval, endpoint):
    query_str= 'query={}[{}]'.format(query, interval)
    print('query {}'.format(query_str))
    return requests.get(url = endpoint, params = query_str).json()


#TODO: Test with Kepler on AWS


def scrape_prometheus_metrics(query, interval, endpoint):
    return query_prometheus(query, interval, endpoint)

#Testing using a Prometheus exporter which monitors itself for metrics
def retrieve_dummy_prometheus_metrics(query, interval, endpoint):
    return scrape_prometheus_metrics(query, interval, endpoint)

def retrieve_and_clean_prometheus_energy_metrics(query, interval, endpoint):
    energy_metrics_dataset = scrape_prometheus_metrics(query, interval, endpoint)
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
        curr_cpu_cycles.append(float(refined_metrics['curr_cpu_cycles']))
        curr_cpu_instructions.append(float(refined_metrics['curr_cpu_instructions']))
        curr_cpu_time.append(float(refined_metrics['curr_cpu_time']))
        
        cpu_architecture.append(refined_metrics['cpu_architecture'])

        curr_energy_in_core.append(float(refined_metrics['curr_energy_in_core']))
        curr_energy_in_dram.append(float(refined_metrics['curr_energy_in_dram']))
        curr_cache_misses.append(float(refined_metrics['curr_cache_misses']))
        curr_resident_memory.append(float(refined_metrics['curr_resident_memory']))

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

