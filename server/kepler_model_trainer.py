import tensorflow as tf
from keras.models import Sequential, load_model
from keras import layers, Model, Input, metrics, optimizers
import numpy as np
import os
import shutil

# Dict to map model type with filepath
model_type_to_filepath = {"core_model": "models/core_model", "dram_model": "models/dram_model"}

core_model_labels = {
                "categorical_string_labels": ["cpu_architecture"], 
                "numerical_lables": ["curr_cpu_cycles", "current_cpu_instructions", "curr_cpu_time"]}

dram_model_labels = {
                    "categorical_string_labels": ["cpu_architecture"],
                    "numerical_lables": ["curr_resident_memory", "curr_cache_misses"]}

cpu_architecture_vocab = ["Sandy Bridge", "Ivy Bridge", "Haswell", "Broadwell", "Sky Lake", "Cascade Lake", "Coffee Lake", "Alder Lake"]


# Generates a Core regression model which predicts curr_energy_in_core given
# numerical features (curr_cpu_cycles, current_cpu_instructions, curr_cpu_time).
# Consumes a tensorflow dataset with no missing data and all required features and 
# target. This Regression model will be saved on server.
def generate_core_regression_model(core_train_dataset: tf.data.Dataset) -> Model:
    all_features_to_modify = []
    input_list = []
    #normalizing numerical features with given dataset
    for numerical_column in core_model_labels["numerical_lables"]:
        new_input = Input(shape=(1,), name=numerical_column) # single list with one constant
        new_normalizer = layers.Normalization(axis=None)
        new_normalizer.adapt(core_train_dataset.map(lambda x, y: x[numerical_column]))
        all_features_to_modify.append(new_normalizer(new_input))
        input_list.append(new_input)
    # encoding categorical feature with given data immediately (can also write this as a new model)
    for categorical_column in core_model_labels["categorical_string_labels"]:
        new_input = Input(shape=(1, ), name=categorical_column, dtype='string') # single list with one constant
        new_int_index = layers.StringLookup(vocabulary=cpu_architecture_vocab)
        #print(new_int_index(tf.constant(["Coffee Lake", "Sandy Bridge", "Haswell", "Coffee Lake", "Haswell"])))
        #new_int_index.adapt(core_train_dataset.map(lambda x, y: x[categorical_column]))
        new_layer = layers.CategoryEncoding(num_tokens=new_int_index.vocabulary_size(), output_mode="one_hot") # no relationship between categories
        all_features_to_modify.append(new_layer(new_int_index(new_input)))
        input_list.append(new_input)
    
    all_features = layers.concatenate(all_features_to_modify)
    single_regression_layer = layers.Dense(units=1, activation='linear', name="linear_regression_layer")(all_features)

    new_linear_model = Model(input_list, single_regression_layer)
    new_linear_model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='mse', metrics=[metrics.RootMeanSquaredError()])
    return new_linear_model
    
    #normalizer = layers.Normalization(axis=-1)
    #normalizer.adapt(np.array(train_features))
    #created a linear regression model with normalized features
    #new_linear_model = Sequential([
    #    normalizer,
    #    layers.Dense(units=1) #len(train_features instead?)
    #])
    #new_linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')
    #return new_linear_model
    pass


def generate_dram_regression_model(dram_train_dataset: tf.data.Dataset) -> Model:
    all_features_to_modify = []
    input_list = []
    #normalizing numerical features with given dataset
    for numerical_column in dram_model_labels["numerical_lables"]:
        new_input = Input(shape=(1,), name=numerical_column) # single list with one constant
        new_normalizer = layers.Normalization(axis=None)
        new_normalizer.adapt(dram_train_dataset.map(lambda x, y: x[numerical_column]))
        all_features_to_modify.append(new_normalizer(new_input))
        input_list.append(new_input)
    # encoding categorical feature with given data immediately (can also write this as a new model)
    for categorical_column in dram_model_labels["categorical_string_labels"]:
        new_input = Input(shape=(1, ), name=categorical_column, dtype='string') # single list with one constant
        new_int_index = layers.StringLookup(vocabulary=cpu_architecture_vocab)
        #new_int_index.adapt(dram_train_dataset.map(lambda x, y: x[categorical_column]))
        new_layer = layers.CategoryEncoding(num_tokens=new_int_index.vocabulary_size(), output_mode='one_hot') # no relationship between categories
        all_features_to_modify.append(new_layer(new_int_index(new_input)))
        input_list.append(new_input)
    
    all_features = layers.concatenate(all_features_to_modify)
    single_regression_layer = layers.Dense(units=1, activation='linear', name="linear_regression_layer")(all_features)

    new_linear_model = Model(input_list, single_regression_layer)

    new_linear_model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='mse', metrics=[metrics.RootMeanSquaredError()])
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
        
#create prometheus scraper

# This function creates a new model and trains it given train_features, train_labels, test_features, test_labels.
# If the desired model already exists, it will be retrained, refitted, evaluated and saved.
# takes dataframe
def train_model_given_data_and_type(train_dataset, validation_dataset, test_dataset, model_type):
    # TODO: Include Demo using excel data - returns evaluation details and coefficients
    filepath, model_exists = return_model_filepath(model_type)
    if filepath is None:
        raise ValueError("Provided Model Type is invalid and/or not included")

    if model_exists:
        new_model = load_model(filepath)
    elif model_type == "core_model":
        new_model = generate_core_regression_model(train_dataset)
    elif model_type == "dram_model":
        new_model = generate_dram_regression_model(train_dataset)
    new_model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
    loss_result, rmse_metric = new_model.evaluate(test_dataset)
    # TODO: Include While loop to ensure loss_result is within acceptable ranges. When to stop?

    # These will be stored in a database containing a timestamp so the most recent loss can be used to determine if the model can be exported
    print("Mean Absolute Error Loss: " + str(loss_result))
    print("Root Mean Squared Error Loss metric: " + str(rmse_metric))
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


# This function archives the SavedModel according to model_type. It returns the filepath to zipped SavedModel
# and the filename (which is just the model type with .zip appended to the end)
def archive_saved_model(model_type):
    filepath, model_exists = return_model_filepath(model_type)
    if filepath is None:
        raise ValueError("Provided Model Type is invalid and/or not included")
    if not model_exists:
        raise FileNotFoundError("The desired trained model is valid, but the model has not been created/saved yet")
    
    shutil.make_archive(filepath, 'zip', filepath)
    # return archived zipped filepath directory
    return os.path.join(os.path.dirname(__file__), 'models/'), model_type + '.zip'


# Returns the weights and bias from the desired model.
def return_model_weights(model_type):
    filepath, model_exists = return_model_filepath(model_type)
    if filepath is not None:
        if model_exists:
            # Retrieve coefficients
            returned_model = load_model(filepath)  
            for layer in returned_model.layers:
                print(layer.name)          
            kernel_matrix = np.ndarray.tolist(returned_model.get_layer(name="linear_regression_layer").get_weights()[0])
            linear_bias = np.ndarray.tolist(returned_model.get_layer(name="linear_regression_layer").get_weights()[1])
            #print(kernel_matrix)
            #print(linear_bias)
            return kernel_matrix, linear_bias
        raise FileNotFoundError("The desired trained model is valid, but the model has not been created/saved yet")
    else:
        raise ValueError("Provided Model Type is invalid and/or not included")

# Test function
def return_model_test_coefficients(model_type):
    filepath, model_exists = return_model_filepath(model_type)
    if filepath is not None:
        if model_exists:
            # Retrieve coefficients
            #returned_model = load_model(filepath)
            #for layer in returned_model.layers: print(layer.get_config(), layer.get_weights())
            #print(returned_model.get_weights())
            #for layer in returned_model.layers:
            #    print(layer.name)
            #    if(layer.name == "string_lookup"):
            #        print(layer.get_vocabulary())
            #    if(layer.name == "concatenate"):
            #        print(layers.Concatenate(layer))
            #    if(layer.name == "linear_regression_layer"):
            #        print(layer.variables)
            #        print(layer.weights)
            return "Placeholder"
        raise FileNotFoundError("The desired trained model is valid, but the model has not been created/saved yet")
    else:
        raise ValueError("Provided Model Type is invalid and/or not included")


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

