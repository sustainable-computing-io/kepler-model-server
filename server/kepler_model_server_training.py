import tensorflow as tf
from keras.models import Sequential, load_model, save_model
from keras import layers

#function to generate a new regression model given features
# regression model will be saved on server
def generate_regression_model(train_features):
    normalizer = layers.Normalization(axis=None)
    normalizer.adapt(train_features)
    #created a linear regression model with normalized features
    new_linear_model = Sequential([ 
        normalizer,
        layers.Dense(units=1)
    ])
    new_linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
    return new_linear_model


def train_core_model(train_features, train_labels, test_features, test_labels):
    # TODO: Include Demo using excel data 
    try:
        returned_model = load_model('models/core_model.h5')
        # after retrieving saved model, call fit
        # returned_model.fit(train_features, train_labels, epochs=100)
        save_model(returned_model, 'models/core_model.h5')

    except IOError:
        # indicates file does not exist
        new_model = generate_regression_model(train_features)
        # new_model.fit(train_features, train_labels, epochs=100)
        save_model(returned_model, 'models/core_model.h5')


def train_dram_model(train_features, train_labels, test_features, test_labels):
    # TODO: Include Demo using dataset in the form of an excel file
    try:
        returned_model = load_model('models/dram_model.h5')
        # after retrieving saved model, train and evaluate
        # returned_model.fit(train_features, train_labels, epochs=100)
        # returned_model.evaluate(test_features, test_labels, verbose=0)
        save_model(returned_model, 'models/dram_model.h5')

    except IOError:
        # indicates file does not exist
        new_model = generate_regression_model(train_features)
        # new_model.fit(train_features, train_labels, epochs=100)
        # new_model.evaluate(test_features, test_labels, verbose=0)
        save_model(returned_model, 'models/dram_model.h5')
