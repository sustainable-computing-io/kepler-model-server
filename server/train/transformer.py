import tensorflow as tf
import numpy as np

class KerasFullPipelineFeatureTransformer():
    def __init__(self, features, dram_model_labels, core_model_labels):
        self.features = features
        self.core_numerical_indexes = [self.features.index(label) for label in core_model_labels['numerical_labels']]
        self.dram_numerical_indexes = [self.features.index(label) for label in dram_model_labels['numerical_labels']]
        self.core_categorical_indexes = [self.features.index(label) for label in core_model_labels['categorical_string_labels']]
        self.dram_categorical_indexes = [self.features.index(label) for label in core_model_labels['categorical_string_labels']]

    def transform(self, x_values):
        inputs = []
        for core_index in self.core_numerical_indexes:
            core_values = x_values[:,core_index:core_index+1].astype(np.float32)
            inputs.append(tf.cast(core_values, tf.float32))
        for core_index in self.core_categorical_indexes:
            inputs.append(tf.cast(x_values[:,core_index:core_index+1], tf.string))
        for dram_index in self.dram_numerical_indexes:
            dram_values = x_values[:,dram_index:dram_index+1].astype(np.float32)
            inputs.append(tf.cast(dram_values, tf.float32))
        for dram_index in self.dram_categorical_indexes:
            inputs.append(tf.cast(x_values[:,dram_index:dram_index+1], tf.string))
        return inputs