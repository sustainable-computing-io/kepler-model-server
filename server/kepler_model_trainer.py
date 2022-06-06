import tensorflow as tf
from keras.models import Sequential, load_model
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

core_names = ['curr_energy_in_core', 'cpu_architecture', 'curr_cpu_cycles', 'current_cpu_instructions', 'curr_cpu_time']
dram_names = ['curr_energy_in_dram', 'cpu_architecture', 'curr_resident_memory', 'curr_cache_misses']

# Generates a new regression model with normalized features
# precondition: data has already been cleaned (no missing data)
# This Regression model will be saved on server
def generate_regression_model(train_features) -> Sequential:
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    #created a linear regression model with normalized features
    new_linear_model = Sequential([
        normalizer,
        layers.Dense(units=1)
    ])
    new_linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
    return new_linear_model


def train_model_given_data_and_filepath(train_features, train_labels, test_features, test_labels, filepath):
    # TODO: Include Demo using excel data - returns evaluation details and coefficients
    if os.path.exists(filepath):
        returned_model = load_model(filepath)
        returned_model.fit(train_features, train_labels, epochs=10, validation_split=0.1)
        results = returned_model.evaluate(test_features, test_labels, verbose=1)
        print("Scaler Loss: " + str(results))
        returned_model.save(filepath, save_format='tf', include_optimizer=True)
    else:
        new_model = generate_regression_model(train_features=train_features)
        new_model.fit(train_features, train_labels, epochs=10, validation_split=0.1)
        results = new_model.evaluate(test_features, test_labels, verbose=1)
        print("Scaler Loss: " + str(results))
        #print(str(new_model.layers))
        new_model.save(filepath, save_format='tf', include_optimizer=True)

    #try:
    #    returned_model = load_model('models/core_model.h5')
        # after retrieving saved model, call fit
        # returned_model.fit(train_features, train_labels, epochs=100)
    #    save_model(returned_model, 'models/core_model.h5')

    #except IOError:
        # indicates file does not exist
    #    new_model = generate_regression_model(train_features)
        # new_model.fit(train_features, train_labels, epochs=100)
    #    save_model(returned_model, 'models/core_model.h5')
    

#def train_dram_model(train_features, train_labels, test_features, test_labels):
    # TODO: Include Demo using dataset in the form of an excel file - returns evaluation details and coefficients    
    #try:
    #    returned_model = load_model('models/dram_model.h5')
        # after retrieving saved model, train and evaluate
        # returned_model.fit(train_features, train_labels, epochs=100)
        # returned_model.evaluate(test_features, test_labels, verbose=0)
    #    save_model(returned_model, 'models/dram_model.h5')

    #except IOError:
        # indicates file does not exist
    #    new_model = generate_regression_model(train_features)
        # new_model.fit(train_features, train_labels, epochs=100)
        # new_model.evaluate(test_features, test_labels, verbose=0)
    #    save_model(returned_model, 'models/dram_model.h5')

# DEMO: Run python kepler_model_trainer.py to test the linear regression models for Dram and Core Energy Consumption using mock data.
# Note that you can also remove the folders in ./models to test creating and saving a new linear regression models.
if __name__ == "__main__":
    #print("Training Commencing\n--------------------------------------------------\n")
    core_dataset = pd.read_csv("core_dataset1.csv", names=core_names, header=None, sep='\t')
    dram_dataset = pd.read_csv('dram_dataset.csv', names=dram_names, header=None, sep='\t')
    # Code to clean dataset should be inserted below
    # One hot encode categorical feature (cpu_architecture) directly onto the dataset.
    # Note it would be better to manage categorical features in the model as a layer. This should be changed when Prometheus Scraper is active. 
    core_dataset = pd.get_dummies(core_dataset, columns=['cpu_architecture'])
    dram_dataset = pd.get_dummies(dram_dataset, columns=['cpu_architecture'])
    #print(dram_dataset.loc[0])
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
    train_model_given_data_and_filepath(core_train_features, core_target_train, core_test_features, core_target_test, "./models/core_model")

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
    train_model_given_data_and_filepath(dram_train_features, dram_target_train, dram_test_features, dram_target_test, "./models/dram_model")

    print("\n--------------------------------------------------\nTraining Completed")