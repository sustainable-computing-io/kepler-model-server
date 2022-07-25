import pandas as pd
import tensorflow as tf
import numpy as np
from kepler_model_trainer import generate_core_regression_model
import os
from keras.callbacks import History 

def create_core_dataset_from_excel_data():
    filepath = os.path.dirname(__file__)
    core_dataframe = pd.read_csv("{}/real_test_datasets/metrics.csv".format(filepath), header=0, sep=',')
    core_dataframe.pop("curr_energy_in_dram")
    core_dataframe.pop("curr_resident_memory")
    core_dataframe.pop("curr_cache_misses")
    train_dataframe, validation_dataframe, test_dataframe = np.split(core_dataframe.sample(frac=1), [int(0.6 * len(core_dataframe)), int(0.8 * len(core_dataframe))])
    train_labels_df = train_dataframe.pop("curr_energy_in_core")
    validation_labels_df = validation_dataframe.pop("curr_energy_in_core")
    test_labels_df = test_dataframe.pop("curr_energy_in_core")

    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_dataframe), train_labels_df))
    train_dataset = train_dataset.shuffle(len(train_dataset)).repeat(4).batch(32)
    validation_dataset = tf.data.Dataset.from_tensor_slices((dict(validation_dataframe), validation_labels_df))
    validation_dataset = validation_dataset.batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_dataframe), test_labels_df))
    test_dataset = test_dataset.batch(32)

    #train_model_given_data_and_type(train_dataset, validation_dataset, test_dataset, "core_model")
    new_model = generate_core_regression_model(train_dataset)
    history = History()
    new_model.fit(train_dataset, epochs=50, validation_data=validation_dataset, callbacks=[history])
    loss_result, r_squared, rmse_metric = new_model.evaluate(test_dataset)
    print("Mean Squared Error: {}".format(loss_result))
    print("R Squared: {}".format(r_squared))
    print("Root Mean Squared Error: {}".format(rmse_metric))

if __name__ == "__main__":
    create_core_dataset_from_excel_data()
    