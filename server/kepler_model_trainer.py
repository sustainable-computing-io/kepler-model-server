import tensorflow as tf
from keras.models import Sequential, load_model
from keras import layers
import numpy as np
import os
import shutil

# Dict to map model type with filepath
model_type_to_filepath = {"core_model": "models/core_model", "dram_model": "models/dram_model"}

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

# Helper function to verify model_type is valid, check whether the model has been created or not,
# and create a filepath to the model.
# Returns type (str, Bool) where str is the filepath to the model_type or None if the model_type
# is invalid, and Bool is True if the model of the desired type was already created and False if the model
# of the desired type has not been created. If str is None, then Bool is False.
def return_model_filepath(model_type):
    if model_type in model_type_to_filepath:
        filepath = os.path.join(os.path.dirname(__file__), model_type_to_filepath[model_type])
        return filepath, os.path.exists(filepath)
        #if os.path.exists(filepath):
        #    return filepath, True
        #return filepath, False
    return None, False
        

def train_model_given_data_and_type(train_features, train_labels, test_features, test_labels, model_type):
    # TODO: Include Demo using excel data - returns evaluation details and coefficients
    filepath, model_exists = return_model_filepath(model_type)
    if filepath is not None:
        if model_exists:
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
    else:
        raise ValueError("Provided Model Type is invalid and/or not included")

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

def archive_saved_model(model_type):
    filepath, model_exists = return_model_filepath(model_type)
    if filepath is None:
        raise ValueError("Provided Model Type is invalid and/or not included")
    if not model_exists:
        raise FileNotFoundError("The desired trained model is valid, but the model has not been created/saved yet")
    
    shutil.make_archive(filepath, 'zip', filepath)
    # return archived zipped filepath directory
    return os.path.join(os.path.dirname(__file__), 'models/'), model_type + '.zip'


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

