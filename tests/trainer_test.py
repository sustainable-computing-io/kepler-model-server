# trainer_test.py

import os
import sys

import sklearn

#################################################################
# import internal src 
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)
#################################################################

from train import load_class
from util import PowerSourceMap
from util.loader import default_train_output_pipeline
from util.train_types import default_trainer_names

from isolator_test import test_isolators, get_isolate_results
from extractor_test import test_extractors, get_extract_results, test_energy_source, get_expected_power_columns, node_info_column

import pandas as pd
import threading

test_trainer_names = default_trainer_names
pipeline_lock = threading.Lock()

def assert_train(trainer, data, energy_components):
    trainer.print_log("assert train")
    node_types = pd.unique(data[node_info_column])
    for node_type in node_types:
        node_type_str = int(node_type)
        node_type_filtered_data = data[data[node_info_column] == node_type]
        X_values = node_type_filtered_data[trainer.features].values
        for component in energy_components:
            try:
                output = trainer.predict(node_type_str, component, X_values)
                assert len(output) == len(X_values), "length of predicted values != features ({}!={})".format(len(output), len(X_values))
            except sklearn.exceptions.NotFittedError:
                pass
            
def process(node_level, feature_group, result, trainer_names=test_trainer_names, energy_source=test_energy_source, power_columns=get_expected_power_columns(), pipeline_name=default_train_output_pipeline):
    energy_components = PowerSourceMap[energy_source]
    train_items = []
    for trainer_name in trainer_names:
        trainer_class = load_class("trainer", trainer_name)
        trainer = trainer_class(energy_components, feature_group, energy_source, node_level=node_level, pipeline_name=pipeline_name)
        trainer.process(result, power_columns, pipeline_lock=pipeline_lock)
        assert_train(trainer, result, energy_components)
        train_items += [trainer.get_metadata()]
    return pd.concat(train_items)

def process_all(extractors=test_extractors, isolators=test_isolators, trainer_names=test_trainer_names, energy_source=test_energy_source, power_columns=get_expected_power_columns(), pipeline_name=default_train_output_pipeline):
    abs_train_list = []
    dyn_train_list = []
    for extractor in extractors:
        extractor_name = extractor.__class__.__name__
        extractor_results = get_extract_results(extractor_name, node_level=True)
        for feature_group, result in extractor_results.items():
            print("Extractor ", extractor_name)
            metadata_df = process(True, feature_group, result, trainer_names=trainer_names, energy_source=energy_source, power_columns=power_columns, pipeline_name=pipeline_name)
            metadata_df['extractor'] = extractor_name
            metadata_df['feature_group'] = feature_group
            abs_train_list += [metadata_df]
            
        for isolator in isolators:
            isolator_name = isolator.__class__.__name__
            isolator_results = get_isolate_results(isolator_name, extractor_name)
            for feature_group, result in isolator_results.items():
                print("Isolator ", isolator_name)
                metadata_df = process(False, feature_group, result, trainer_names=trainer_names, energy_source=energy_source, power_columns=power_columns, pipeline_name=pipeline_name)
                metadata_df['extractor'] = extractor_name
                metadata_df['isolator'] = isolator_name
                metadata_df['feature_group'] = feature_group
                dyn_train_list += [metadata_df]
    abs_train_df = pd.concat(abs_train_list)
    dyn_train_df = pd.concat(dyn_train_list)
    return abs_train_df, dyn_train_df


        
if __name__ == '__main__':
    focus_columns = ['model_name', 'mae']
    abs_train_df, dyn_train_df = process_all()
    print("Node-level train results:")
    print(abs_train_df.set_index(['extractor', 'feature_group'])[focus_columns].sort_values(by=['mae'], ascending=True))
    print("Container-level train results:")
    print(dyn_train_df.set_index(['extractor', 'isolator', 'feature_group'])[focus_columns].sort_values(by=['mae'], ascending=True))