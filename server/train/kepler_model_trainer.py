import tensorflow as tf
from keras.models import load_model
from keras import layers, Model, Input, metrics, optimizers
import numpy as np
import os
import shutil
from keras import backend as K
from keras.callbacks import History

from train_types import CATEGORICAL_LABEL_TO_VOCAB
# Dict to map model type with filepath
model_type_to_filepath = {"core": "/models/core_model", "dram": "/models/dram_model"}

core_model_labels = {
                "numerical_labels": ["curr_cpu_cycles", "curr_cpu_instr", "curr_cpu_time"],
                "categorical_string_labels": ["cpu_architecture"] }

dram_model_labels = {
                    "numerical_labels": ["container_memory_working_set_bytes", "curr_cache_miss"],
                    "categorical_string_labels": ["cpu_architecture"]}




def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Generates a Core regression model which predicts curr_energy_in_core given
# numerical features (curr_cpu_cycles, curr_cpu_instructions, curr_cpu_time).
# Consumes a tensorflow dataset with no missing data and all required features and 
# target. This Regression model will be saved on server.
def generate_core_regression_model(core_train_dataset: tf.data.Dataset) -> Model:
    prefix = 'core_'
    all_features_to_modify = []
    input_list = []
    #normalizing numerical features with given dataset
    for numerical_column in core_model_labels["numerical_labels"]:
        new_input = Input(shape=(1,), name=prefix + numerical_column) # single list with one constant
        new_normalizer_name = "normalization_" + numerical_column
        new_normalizer = layers.Normalization(axis=None, name=new_normalizer_name)
        new_normalizer.adapt(core_train_dataset.map(lambda x, y: x[prefix + numerical_column]))
        all_features_to_modify.append(new_normalizer(new_input))
        input_list.append(new_input)
    # encoding categorical feature with given data immediately (can also write this as a new model)
    for categorical_column in core_model_labels["categorical_string_labels"]:
        new_input = Input(shape=(1, ), name=prefix + categorical_column, dtype='string') # single list with one constant
        new_int_index = layers.StringLookup(vocabulary=CATEGORICAL_LABEL_TO_VOCAB[categorical_column], num_oov_indices=0)
        #new_int_index.adapt(core_train_dataset.map(lambda x, y: x[categorical_column]))
        new_layer = layers.CategoryEncoding(num_tokens=new_int_index.vocabulary_size(), output_mode='one_hot') # no relationship between categories
        all_features_to_modify.append(new_layer(new_int_index(new_input)))
        input_list.append(new_input)
    
    all_features = layers.concatenate(all_features_to_modify)
    single_regression_layer = layers.Dense(units=1, activation='linear', name="core_linear_regression_layer")(all_features)
    new_linear_model = Model(input_list, single_regression_layer)
    new_linear_model.compile(optimizer=optimizers.Adam(learning_rate=0.5), loss='mae', metrics=[coeff_determination, metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()])
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


def generate_dram_regression_model(dram_train_dataset: tf.data.Dataset) -> Model:
    prefix = 'dram_'
    all_features_to_modify = []
    input_list = []
    #normalizing numerical features with given dataset
    for numerical_column in dram_model_labels["numerical_labels"]:
        new_input = Input(shape=(1,), name=prefix + numerical_column) # single list with one constant
        new_normalizer_name = "normalization_" + numerical_column
        new_normalizer = layers.Normalization(axis=None, name=new_normalizer_name)
        new_normalizer.adapt(dram_train_dataset.map(lambda x, y: x[prefix + numerical_column]))
        all_features_to_modify.append(new_normalizer(new_input))
        input_list.append(new_input)
    # encoding categorical feature with given data immediately (can also write this as a new model)
    for categorical_column in dram_model_labels["categorical_string_labels"]:
        new_input = Input(shape=(1, ), name=prefix + categorical_column, dtype='string') # single list with one constant
        new_int_index = layers.StringLookup(vocabulary=CATEGORICAL_LABEL_TO_VOCAB[categorical_column], num_oov_indices=0)
        #new_int_index.adapt(dram_train_dataset.map(lambda x, y: x[categorical_column]))
        new_layer = layers.CategoryEncoding(num_tokens=new_int_index.vocabulary_size(), output_mode='one_hot') # no relationship between categories
        all_features_to_modify.append(new_layer(new_int_index(new_input)))
        input_list.append(new_input)
    
    all_features = layers.concatenate(all_features_to_modify)
    single_regression_layer = layers.Dense(units=1, activation='linear', name="dram_linear_regression_layer")(all_features)

    new_linear_model = Model(input_list, single_regression_layer)

    new_linear_model.compile(optimizer=optimizers.Adam(learning_rate=0.5), loss='mae', metrics=[coeff_determination, metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()])
    return new_linear_model


# Helper function to verify model_type is valid, check whether the model has been created or not,
# and create a filepath to the model.
# Returns type (str, Bool) where str is the filepath to the model_type or None if the model_type
# is invalid, and Bool is True if the model of the desired type was already created and False if the model
# of the desired type has not been created. If str is None, then Bool is False.
def return_model_filepath(model_type):
    if model_type in model_type_to_filepath:
        filepath = os.path.dirname(__file__) + model_type_to_filepath[model_type]
        return filepath, os.path.exists(filepath)
        #if os.path.exists(filepath):
        #    return filepath, True
        #return filepath, False
    return None, False
        
# This function creates a new model and trains it given train_features, train_labels, test_features, test_labels.
# If the desired model already exists, it will be retrained, refitted, evaluated and saved.
# takes dataframe
def train_model_given_data_and_type(new_model, train_dataset, validation_dataset, test_dataset, model_type):
    # TODO: Include Demo using excel data - returns evaluation details and coefficients
    history = History()
    new_model.fit(train_dataset, epochs=5, validation_data=validation_dataset, callbacks=[history])
    loss_result, r_squared, rmse_metric, mae_metric = new_model.evaluate(test_dataset)
    # TODO: Include While loop to ensure loss_result is within acceptable ranges (Save only if loss is acceptable)

    print("Mean Squared Error Loss: " + str(loss_result))
    print("Root Mean Squared Error Loss metric: " + str(rmse_metric))
    print("Mean Absolute Error Loss metric: " + str(mae_metric))
    print("Correlation of Determination:" + str(r_squared))
    print("Training MSE Results per epoch: {}".format(history.history['loss']))
    print("Training RMSE Results per epoch: {}".format(history.history['root_mean_squared_error']))
    print("Training Correlation Coefficient per epoch: {}".format(history.history['coeff_determination']))
    print("Validation MSE per epoch: {}".format(history.history['val_loss']))
    print("Validation RMSE per epoch: {}".format(history.history['val_root_mean_squared_error']))
    print("Validation Correlation Coefficient per epoch: {}".format(history.history['val_coeff_determination']))
    # new_model.save(filepath, save_format='tf', include_optimizer=True)
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
    return new_model, loss_result, rmse_metric, mae_metric


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
    return os.path.dirname(__file__) + '/models/', model_type + '.zip'


def create_numerical_labels_weights_relation(numerical_labels, numerical_weights, mean_variance_for_each_label):
    return {numerical_labels[i]: {'weight': weight, "mean": mean_variance_for_each_label[i][0], "variance": mean_variance_for_each_label[i][1]} for i, weight in enumerate(numerical_weights)}


def create_categorical_vocab_weights_relation(categorical_labels, list_of_all_categorical_labels_vocab, list_of_all_categorical_labels_weights):
    return {label: {list_of_all_categorical_labels_vocab[i][j]: {'weight': list_of_all_categorical_labels_weights[i][j]} for j in range(len(list_of_all_categorical_labels_vocab[i]))} for i, label in enumerate(categorical_labels)}

# Returns the weights and bias from the desired model.
def return_model_weights(model_type):
    filepath, model_exists = return_model_filepath(model_type)
    if filepath is not None:
        if model_exists:
            # Retrieve Model
            returned_model = load_model(filepath, custom_objects={'coeff_determination': coeff_determination})
            #TODO: In future, create a dict to divide the model algorithm type (Linear, NN, etc.) for better
            # abstraction and to avoid repeat code. Currently, all models are Regression with Normalized Numerical
            # Labels and String Categorical Labels. All Models have the same name for the regression layer and same naming
            # convention for normalized preprocessing layers.

            # Retrieve Coefficients
            kernel_matrix = returned_model.get_layer(name=model_type +"_linear_regression_layer").get_weights()[0]
            bias = returned_model.get_layer(name=model_type +"linear_regression_layer").get_weights()[1][0].item()
            weights_list = np.ndarray.tolist(np.squeeze(kernel_matrix))
            print(weights_list)
            print(bias)
            if model_type == "core":
                # Retrieve Numerical Labels for Core Model
                numerical_labels = core_model_labels["numerical_labels"]
                # Retrieve Categorical Labels for Core Model
                categorical_labels = core_model_labels["categorical_string_labels"]
            if model_type == "dram":
                # Retrieve Numerical Labels for Core Model
                numerical_labels = dram_model_labels["numerical_labels"]
                # Retrieve Categorical Labels for Core Model
                categorical_labels = dram_model_labels["categorical_string_labels"]
                
            # Numerical Weights
            start_index = len(numerical_labels)
            numerical_weights = weights_list[0:start_index]
            # Normalization Weights
            normalization_weights = []
            for numerical_label in numerical_labels:
                name = "normalization_" + numerical_label
                normalization_weight = np.float_(returned_model.get_layer(name=name).get_weights())
                normalization_weights.append(normalization_weight)

            # Numerical Label and Weights Dict
            numerical_weights_dict = create_numerical_labels_weights_relation(numerical_labels, numerical_weights, normalization_weights)
            
            # Categorical Weights
            list_of_all_categorical_labels_vocab = []
            list_of_all_categorical_labels_weights = []
            for categorical_label in categorical_labels:
                categorical_label_vocab = CATEGORICAL_LABEL_TO_VOCAB[categorical_label]
                list_of_all_categorical_labels_vocab.append(categorical_label_vocab)
                end_index = start_index + len(categorical_label_vocab)
                categorical_label_weights = weights_list[start_index:end_index]
                list_of_all_categorical_labels_weights.append(categorical_label_weights)
                start_index = end_index
            
            #Categorical Label and Weights Dict
            categorical_weights_dict = create_categorical_vocab_weights_relation(categorical_labels, list_of_all_categorical_labels_vocab, list_of_all_categorical_labels_weights)
            # Combine all features and weights into a single dictionary
            final_dict = {"All_Weights": {"Numerical_Variables": numerical_weights_dict, "Categorical_Variables": categorical_weights_dict, "Bias_Weight": bias} }
            return final_dict
            
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

def create_prometheus_core_dataset(cpu_architecture, curr_cpu_cycles, curr_cpu_instructions, curr_cpu_time, curr_energy_in_core): #query: str, length: int, endpoint: str
    prefix = 'core_'
    # Create Desired Datasets for Core Model
    features_dict_core = {prefix + 'cpu_architecture': cpu_architecture, prefix + 'curr_cpu_cycles': curr_cpu_cycles, prefix + 'curr_cpu_instr': curr_cpu_instructions, prefix + 'curr_cpu_time': curr_cpu_time}
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
    prefix = 'dram_'
    # Create Desired Dataset for Dram Model
    features_dict_dram = {prefix + 'cpu_architecture': cpu_architecture, prefix + 'container_memory_working_set_bytes': curr_resident_memory, prefix + 'curr_cache_miss': curr_cache_misses}
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

def merge_model(core_model, dram_model):
    # concat_layer = layers.Concatenate([core_model, dram_model])
    # output = layers.Lambda(tf.reduce_sum, arguments=dict(axis=1))(concat_layer)
    output = layers.Add()([core_model.output, dram_model.output])
    model = Model(inputs=core_model.inputs + dram_model.inputs, outputs=output)
    return model
