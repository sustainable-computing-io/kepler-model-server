import pandas as pd
from server.kepler_model_trainer import train_model_given_data_and_type
import os
import tensorflow as tf
import numpy as np


# Test Names
core_names = ['curr_energy_in_core', 'cpu_architecture', 'curr_cpu_cycles', 'current_cpu_instructions', 'curr_cpu_time']
dram_names = ['curr_energy_in_dram', 'cpu_architecture', 'curr_resident_memory', 'curr_cache_misses']

# Function to convert test csv files to dataset. Modified from the tensorflow keras website. This will not be
# needed for the prometheus data pipeline
def convert_to_dataset(dataframe, target):
    dataframe = dataframe.copy()
    target_dataframe = dataframe.pop(target)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), target_dataframe))
    ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(1)
    return ds

def test_core_regression_model():
    core_dataframe = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_regression_datasets/core_dataset.csv"), names=core_names, header=None, sep='\t')
    # Code to clean dataset should be inserted below
    # One hot encode categorical feature (cpu_architecture) directly onto the dataset.
    # Note it would be better to manage categorical features in the model as a layer. This should be changed when Prometheus Scraper is active. 
    #core_dataset = pd.get_dummies(core_dataset, columns=['cpu_architecture'], prefix="cpu_architecture_")
    # Setup Core Train, Core Test, and Core Validation
    train_dataframe, validation_dataframe, test_dataframe = np.split(core_dataframe.sample(frac=1), [int(0.6 * len(core_dataframe)), int(0.8 * len(core_dataframe))])

    # Convert dataframe to easy to use tensorflow datasets
    train_dataset = convert_to_dataset(train_dataframe, "curr_energy_in_core")
    test_dataset = convert_to_dataset(test_dataframe, "curr_energy_in_core")
    validation_dataset = convert_to_dataset(validation_dataframe, "curr_energy_in_core")
    
    # Train core with the linear regression model    
    train_model_given_data_and_type(train_dataset, validation_dataset, test_dataset, "core_model")


def test_dram_regression_model():
    dram_dataframe = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_regression_datasets/dram_dataset.csv"), names=dram_names, header=None, sep='\t')
    # Code to clean dataset should be inserted below
    # One hot encode categorical feature (cpu_architecture) directly onto the dataset.
    # Note it would be better to manage categorical features in the model as a layer. This should be changed when Prometheus Scraper is active. 
    #dram_dataset = pd.get_dummies(dram_dataset, columns=['cpu_architecture'], prefix="cpu_architecture_")
    #print(dram_dataset.loc[2])
    train_dataframe, validation_dataframe, test_dataframe = np.split(dram_dataframe.sample(frac=1), [int(0.6 * len(dram_dataframe)), int(0.8 * len(dram_dataframe))])
    
    # Convert dataframe to easy to use tensorflow datasets
    train_dataset = convert_to_dataset(train_dataframe, "curr_energy_in_dram")
    test_dataset = convert_to_dataset(test_dataframe, "curr_energy_in_dram")
    validation_dataset = convert_to_dataset(validation_dataframe, "curr_energy_in_dram")

    # Train dram with the linear regression model
    train_model_given_data_and_type(train_dataset, validation_dataset, test_dataset, "dram_model")
    
    # Setup Dram Train and Dram Test
    #dram_train = dram_dataset.sample(frac=0.8)
    #dram_test = dram_dataset.drop(dram_train.index)

    # Create trained and test features for Dram (to avoid distorting original data).
    #dram_train_features = dram_train.copy()
    #dram_test_features = dram_test.copy()

    # Create targets for trained and test Dram
    #dram_target_train = dram_train_features.pop('curr_energy_in_dram')
    #dram_target_test = dram_test_features.pop('curr_energy_in_dram')
    #print(dram_train_features.loc[0])
    

# DEMO: Run python kepler_regression_model_tests.py (python -m tests.kepler_regression_model_tests) to test the linear regression models for Dram and Core Energy Consumption using mock data.
# Note that you can also remove the folders in /server/models to test creating and saving new linear regression models.
if __name__ == "__main__":
    print("Training Commencing\n--------------------------------------------------\n")
    test_core_regression_model()
    test_dram_regression_model()
    print("\n--------------------------------------------------\nTraining Completed")