import pandas as pd
from server.kepler_model_trainer import train_model_given_data_and_type
import os

# Test Names
core_names = ['curr_energy_in_core', 'cpu_architecture', 'curr_cpu_cycles', 'current_cpu_instructions', 'curr_cpu_time']
dram_names = ['curr_energy_in_dram', 'cpu_architecture', 'curr_resident_memory', 'curr_cache_misses']

def test_core_regression_model():
    core_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_regression_datasets/core_dataset.csv"), names=core_names, header=None, sep='\t')
    # Code to clean dataset should be inserted below
    # One hot encode categorical feature (cpu_architecture) directly onto the dataset.
    # Note it would be better to manage categorical features in the model as a layer. This should be changed when Prometheus Scraper is active. 
    core_dataset = pd.get_dummies(core_dataset, columns=['cpu_architecture'])
    #print(core_dataset.loc[0])
    # Setup Core Train and Core Test
    core_train = core_dataset.sample(frac=0.8)
    core_test = core_dataset.drop(core_train.index)

    # Create trained and test features for Core (to avoid distorting original data).
    core_train_features = core_train.copy()
    core_test_features = core_test.copy()

    # Create targets for trained and test Core.
    core_target_train = core_train_features.pop('curr_energy_in_core')
    core_target_test = core_test_features.pop('curr_energy_in_core')
    
    #print(core_train_features)

    # Train core with the linear regression model    
    train_model_given_data_and_type(core_train_features, core_target_train, core_test_features, core_target_test, "CoreEnergyConsumption")


def test_dram_regression_model():
    dram_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_regression_datasets/dram_dataset.csv"), names=dram_names, header=None, sep='\t')
    # Code to clean dataset should be inserted below
    # One hot encode categorical feature (cpu_architecture) directly onto the dataset.
    # Note it would be better to manage categorical features in the model as a layer. This should be changed when Prometheus Scraper is active. 
    dram_dataset = pd.get_dummies(dram_dataset, columns=['cpu_architecture'])
    #print(dram_dataset.loc[2])
    
    # Setup Dram Train and Dram Test
    dram_train = dram_dataset.sample(frac=0.8)
    dram_test = dram_dataset.drop(dram_train.index)

    # Create trained and test features for Dram (to avoid distorting original data).
    dram_train_features = dram_train.copy()
    dram_test_features = dram_test.copy()

    # Create targets for trained and test Dram
    dram_target_train = dram_train_features.pop('curr_energy_in_dram')
    dram_target_test = dram_test_features.pop('curr_energy_in_dram')
    #print(dram_train_features.loc[0])
    # Train dram with the linear regression model
    train_model_given_data_and_type(dram_train_features, dram_target_train, dram_test_features, dram_target_test, "DramEnergyConsumption")


# DEMO: Run python kepler_regression_model_tests.py (python -m tests.kepler_regression_model_tests) to test the linear regression models for Dram and Core Energy Consumption using mock data.
# Note that you can also remove the folders in /server/models to test creating and saving new linear regression models.
if __name__ == "__main__":
    print("Training Commencing\n--------------------------------------------------\n")
    test_core_regression_model()
    test_dram_regression_model()
    print("\n--------------------------------------------------\nTraining Completed")