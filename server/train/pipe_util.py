import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
import tensorflow as tf
from keras.models import load_model
from keras import layers, Model, Input, metrics, optimizers
import numpy as np
from keras import backend as K
from keras.callbacks import History
import xgboost as xgb
import pandas as pd
 

from train_types import CATEGORICAL_LABEL_TO_VOCAB, XGBoostRegressionTrainType, XGBoostMissingModelXOrModelDescException, XGBoostModelFeatureOrLabelIncompatabilityException
# Dict to map model type with filepath
model_type_to_filepath = {"core": "/models/core_model", "dram": "/models/dram_model"}

core_model_labels = {
                "numerical_labels": ["cpu_cycles", "cpu_instr", "cpu_time"],
                "categorical_string_labels": [] }

dram_model_labels = {
                    "numerical_labels": ["cache_miss"],
                    "categorical_string_labels": []}


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# Generates a Core regression model which predicts energy_in_core given
# numerical features (cpu_cycles, cpu_instructions, cpu_time).
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
    
    if len(all_features_to_modify) > 1:
        all_features = layers.concatenate(all_features_to_modify)
    else:
        all_features = all_features_to_modify[0]
    
    single_regression_layer = layers.Dense(units=1, activation='linear', name="core_linear_regression_layer")(all_features)
    new_linear_model = Model(input_list, single_regression_layer)
    new_linear_model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='mae', metrics=[coeff_determination, metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()])
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
    
    if len(all_features_to_modify) > 1:
        all_features = layers.concatenate(all_features_to_modify)
    else:
        all_features = all_features_to_modify[0]

    single_regression_layer = layers.Dense(units=1, activation='linear', name="dram_linear_regression_layer")(all_features)

    new_linear_model = Model(input_list, single_regression_layer)

    new_linear_model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='mae', metrics=[coeff_determination, metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()])
    return new_linear_model

# This function creates a new model and trains it given train_features, train_labels, test_features, test_labels.
# If the desired model already exists, it will be retrained, refitted, evaluated and saved.
# takes dataframe
def train_model_given_data_and_type(new_model, train_dataset, validation_dataset, test_dataset, model_type):
    # TODO: Include Demo using excel data - returns evaluation details and coefficients
    history = History()
    new_model.fit(train_dataset, epochs=500, validation_data=validation_dataset, callbacks=[history])
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


def create_numerical_labels_weights_relation(numerical_labels, numerical_weights, mean_variance_for_each_label):
    return {numerical_labels[i]: {'weight': weight, "mean": mean_variance_for_each_label[i][0], "variance": mean_variance_for_each_label[i][1]} for i, weight in enumerate(numerical_weights)}


def create_categorical_vocab_weights_relation(categorical_labels, list_of_all_categorical_labels_vocab, list_of_all_categorical_labels_weights):
    return {label: {list_of_all_categorical_labels_vocab[i][j]: {'weight': list_of_all_categorical_labels_weights[i][j]} for j in range(len(list_of_all_categorical_labels_vocab[i]))} for i, label in enumerate(categorical_labels)}

# Returns the weights and bias from the desired model.
def return_model_weights(filepath, model_type):
    # Retrieve Model
    returned_model = load_model(filepath, custom_objects={'coeff_determination': coeff_determination})
    #TODO: In future, create a dict to divide the model algorithm type (Linear, NN, etc.) for better
    # abstraction and to avoid repeat code. Currently, all models are Regression with Normalized Numerical
    # Labels and String Categorical Labels. All Models have the same name for the regression layer and same naming
    # convention for normalized preprocessing layers.

    # Retrieve Coefficients
    kernel_matrix = returned_model.get_layer(name=model_type +"_linear_regression_layer").get_weights()[0]
    bias = returned_model.get_layer(name=model_type +"_linear_regression_layer").get_weights()[1][0].item()
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

def create_prometheus_core_dataset(cpu_architecture, cpu_cycles, cpu_instructions, cpu_time, energy_in_core): #query: str, length: int, endpoint: str
    prefix = 'core_'
    # Create Desired Datasets for Core Model
    features_dict_core = {prefix + 'cpu_architecture': cpu_architecture, prefix + 'cpu_cycles': cpu_cycles, prefix + 'cpu_instr': cpu_instructions, prefix + 'cpu_time': cpu_time}
    refined_dataset_core = tf.data.Dataset.from_tensor_slices((features_dict_core, energy_in_core))

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


def create_prometheus_dram_dataset(cpu_architecture, cache_misses, resident_memory, energy_in_dram):
    prefix = 'dram_'
    # Create Desired Dataset for Dram Model
    features_dict_dram = {prefix + 'cpu_architecture': cpu_architecture, prefix + 'container_memory_working_set_bytes': resident_memory, prefix + 'cache_miss': cache_misses}
    refined_dataset_dram = tf.data.Dataset.from_tensor_slices((features_dict_dram, energy_in_dram))
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


# XGBoost (Gradient Boosting Regressor) Base Model Generation
class XGBoostRegressionModelGenerationPipeline():
    """A class used to handle XGBoost Regression Model Incremental Training. This class currently only handles numerical features.

    ...

    Attributes
    ----------
    TODO

    Methods
    -------
    TODO

    """

    feature_names: List[str]
    label_names: List[str]
    save_location: str
    model_name: str

    def __init__(self, feature_names_in_order: List[str], label_names_in_order: List[str], save_location: str, model_name: str) -> None:
        # model data will be generated consistently using the list of feature names and labels (Order does not matter)

        self.feature_names = feature_names_in_order.copy().sort()
        self.label_names = label_names_in_order.copy().sort()
        self.save_location = save_location
        self.model_name = model_name
        self.model_filename = model_name + '.model'
        self.model_desc = 'model_desc.json'


    @staticmethod
    def _generate_base_model() -> xgb.XGBRegressor:
        # n_estimators, max_depth, eta (learning rate)
        return xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1)


    def _generate_model_data_filepath(self) -> str:
        return os.path.join(self.save_location, self.model_name + "_package")

            
    def model_exists(self) -> bool:
        filename_path = self._generate_model_data_filepath()
        return os.path.exists(os.path.join(filename_path, self.model_filename))


    def model_json_data_exists(self) -> bool:
        filename_path = self._generate_model_data_filepath()
        return os.path.exists(os.path.join(filename_path, self.model_desc))


    def retrieve_all_model_data(self) -> Tuple[Optional[xgb.XGBRegressor], Optional[Dict[Any, Any]]]:
        # Note that when generating base model, it does not need to contain default hyperparameters if it will just be
        # used for prediction 
        # Returns model and model_desc
        filename_path = self._generate_model_data_filepath()
        new_model = self._generate_base_model()
        if (not self.model_exists()) ^ (not self.model_json_data_exists()):
            raise XGBoostMissingModelXOrModelDescException(missing_model=self.model_exists(), missing_model_desc=self.model_json_data_exists())
        if self.model_exists() and self.model_json_data_exists():
            new_model.load_model(os.path.join(filename_path, self.model_filename))
            with open(os.path.join(filename_path, self.model_desc), 'r') as f:
                json_data = json.load(f)
            if json_data['feature_names'] != self.feature_names or json_data['label_names'] != self.label_names:
                raise XGBoostModelFeatureOrLabelIncompatabilityException(json_data['feature_names'], json_data['label_names'], self.feature_names, self.label_names)
            return new_model, json_data        
        return None, None


    def _save_model(self, model: xgb.XGBRegressor, model_desc: Dict[Any, Any]) -> None:
        filename_path = self._generate_model_data_filepath()
        model.save_model(os.path.join(filename_path, self.model_filename))
        if "feature_names" not in model_desc:
            model_desc["feature_names"] = self.feature_names
        if "label_names" not in model_desc:
            model_desc["label_names"] = self.label_names
        print(model_desc)
        with open(os.path.join(filename_path, self.model_desc), "w") as f:
            json.dump(model_desc, f)


    def _clean_model_data(self, model_data: pd.DataFrame) -> pd.DataFrame:
        # Create df with relevant feature names and label names
        new_df = pd.DataFrame()
        for feature in self.feature_names:
            new_df[feature] = model_data[feature]
        for label in self.label_names:
            new_df[label] = model_data[label]

        new_df.dropna(inplace=True)
        return new_df
        

    def train(self, train_type: XGBoostRegressionTrainType, model_data: pd.DataFrame) -> None:
        # train_type must contain feature_names columns and label_names columns
        retrieved_model, retrieved_model_desc = self.retrieve_all_model_data()
        all_model_data_exists = retrieved_model is not None and retrieved_model_desc is not None
        cleaned_model_data = self._clean_model_data(model_data)

        # Either cross evaluate or perform simple test and train (Include more training styles if necessary in the future)
        # Note: We can use GridSearchCV to acquire the best hyperparameters (possible automatic feature in the future)
        if train_type == XGBoostRegressionTrainType.TrainTestSplitFit:
            self.__perform_train_test_split(all_model_data_exists, cleaned_model_data)
        elif train_type == XGBoostRegressionTrainType.KFoldCrossValidation:
            self.__perform_kfold_train(all_model_data_exists, cleaned_model_data)
    
        
    def __perform_train_test_split(self, all_model_data_exists: bool, ready_model_data: pd.DataFrame) -> None:
        # Generate new model 
        new_model = self._generate_base_model()
        X = ready_model_data.loc[:,self.feature_names].values
        y = ready_model_data.loc[:,self.label_names].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
        if all_model_data_exists:
            # fit old model with new data
            old_model_filepath = os.path.join(self._generate_model_data_filepath, self.model_filename)
            new_model.fit(X_train, y_train, xgb_model=old_model_filepath)
        else:
            # fit model from scratch
            new_model.fit(X_train, y_train)
        # Evaluate Results and store in model_desc.json
        y_predictions = new_model.predict(X_test)
        
        results = [value for value in y_predictions]

        rmse_res = mean_squared_error(y_test, results, squared=False)
        mae_res = mean_absolute_error(y_test, results)
        r2_res = r2_score(y_test, results)
        # Results
        print(f"rmse: {rmse_res}")
        print(f"r2: {r2_res}")
        print(f"mae: {mae_res}")
        # TODO: Determine acceptable rmse, and r2 values?
        # Note that this feature could also be implemented in kepler-models
        # Generate new model_desc.json
        model_data = {
            "timestamp": datetime.datetime.now().timestamp(),
            "train_type": "fit_train_and_predict_with_test",
            "rmse": rmse_res,
            "r2": r2_res,
            "mae": mae_res,
        }
        # Save Model and model_desc.json
        self._save_model(new_model, model_data)


    def __perform_kfold_train(self, all_model_data_exists: bool, ready_model_data: pd.DataFrame) -> None:
        # Generate new model 
        new_model = self._generate_base_model()
        X = ready_model_data.loc[:,self.feature_names].values
        y = ready_model_data.loc[:,self.label_names].values

        # In the future, contemplate including RepeatedKFold
        kFoldCV = RepeatedKFold(n_splits=10, n_repeats=3, random_state=7)
        old_model_filepath = os.path.join(self._generate_model_data_filepath, self.model_filename)

        if all_model_data_exists:
            params = {
                "xgb_model": old_model_filepath,
            }
            cv_scores = cross_val_score(estimator=new_model, X=X, y=y, scoring="neg_mean_absolute_error", cv=kFoldCV, fit_params=params)
            cv_scores = np.absolute(cv_scores)
        else:
            cv_scores = cross_val_score(estimator=new_model, X=X, y=y, scoring="neg_mean_absolute_error", cv=kFoldCV)
            cv_scores = np.absolute(cv_scores)

        print(f"average_mae: {cv_scores.mean()}")
        print(f"std: {cv_scores.std()}")
        model_data = {
            "timestamp": datetime.datetime.now().timestamp(),
            "train_type": "cross_val_score_Repeated3KFold",
            "average_mae": cv_scores.mean(),
            "std": cv_scores.std(),
        }
        # TODO: Determine acceptable average_mae?
        # Note that this feature could also be implemented in kepler-models

        if all_model_data_exists:
            new_model.fit(X, y, xgb_model=old_model_filepath)
        else:
            # If Model is acceptable, fit it from scratch
            new_model.fit(X, y)

        # Save Model and model_desc.json
        self._save_model(new_model, model_data)
        
    # Receives list of features and returns a list of predictions in order
    # Return None if no available model
    def predict(self, input_values: List[List[float]]) -> Optional[List[float]]:
        retrieved_model, _ = self.retrieve_all_model_data()
        predicted_results = []
        if retrieved_model is not None:
            for feature_input in input_values:
                predict_features = np.asarray([feature_input])
                predicted_result = retrieved_model.predict(predict_features)
                predicted_results.append(predicted_result)
            return predicted_results
        