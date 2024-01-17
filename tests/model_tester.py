# deprecated
import pandas as pd

import os
import sys
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error
from prom_test import get_query_results

src_path = os.path.join(os.path.dirname(__file__), '../src')
train_path = os.path.join(os.path.dirname(__file__), '../src/train')
profile_path = os.path.join(os.path.dirname(__file__), '../src/profile')
util_path = os.path.join(os.path.dirname(__file__), 'util')

sys.path.append(src_path)
sys.path.append(train_path)
sys.path.append(profile_path)
sys.path.append(util_path)

# model_tester.py
# to get the test result across different train/test data set

from train import DefaultExtractor
from profile import profile_process, get_min_max_watt
from train_types import ModelOutputType, PowerSourceMap
from train.isolator.train_isolator import get_background_containers, isolate_container
from offline_trainer_test import get_pipeline_name, isolators, offline_trainer_output_path
from estimator.load import load_model

from prom_types import prom_responses_to_results, TIMESTAMP_COL

extractor = DefaultExtractor()
 

def list_subfolder(top_path):
    return [f for f in os.listdir(top_path) if not os.path.isdir(os.path.join(top_path, f))]


# return mae, mse
def compute_error(predicted_power, actual_powers):
    mse = mean_squared_error(actual_powers, predicted_power)
    mae = mean_absolute_error(actual_powers, predicted_power)
    return mae, mse

# return model, metadata
def process(train_dataset_name, test_dataset_name, target_path):
    idle_data = get_query_results(save_name="idle")
    background_containers = get_background_containers(idle_data)
    profiles = profile_process(idle_data)
    test_data = prom_responses_to_results()
    node_types, _ = extractor.get_node_types(idle_data)
    if node_types is None:
        node_type = "1" # default node type
    else:
        node_type = node_types[0] # limit only one node type in single data set

    # find best_ab
    best_abs_model = find_best_abs_model()

    for isolator, _ in isolators.items():
        print("Isolator: ", isolator)
        pipeline_name = get_pipeline_name(train_dataset_name, isolator)
        save_path = os.path.join(target_path, pipeline_name)
        for energy_source in list_subfolder(save_path):
            source_path = os.path.join(save_path, energy_source)
            # perform AbsPower first to get the best abs model
            for output_type in [ModelOutputType.AbsPower, ModelOutputType.DynPower]:
                model_output_path = os.path.join(source_path, output_type.name)
                for feature_group in list_subfolder(model_output_path):
                    feature_path = os.path.join(source_path, feature_group)
                    model_paths = [os.path.join(feature_path, f) for f in list_subfolder(feature_path)]
                    for model_path in model_paths:
                        model = load_model(model_path)
                        energy_components = PowerSourceMap[energy_source]
                        extracted_data, power_columns, _, _ = extractor.extract(test_data, energy_components, feature_group, energy_source, node_level=False)
                        feature_columns = [col for col in extracted_data.columns if col not in power_columns]
                        if not model.feature_check(feature_columns):
                            print("model {} ({}/{}/{})is not valid to test".format(model.name, energy_source, output_type.name, feature_group))
                            continue
                        if output_type == ModelOutputType.AbsPower:
                            data_with_prediction = extracted_data.copy()
                            predicted_data = model.get_power(extracted_data)
                        else:
                            target_data, background_data = isolate_container(extracted_data, background_containers)
                            data_with_prediction = target_data.copy()
                            predicted_data = model.get_power(target_data)
                            abs_model = best_abs_model[energy_source]
                            predicted_background_power = abs_model.get_power(background_data)
                            predicted_background_dynamic_power = model.get_power(background_data)
                        
                        # for each energy_component 
                        for energy_component, values in predicted_data.items():
                            item = {
                                "train_dataset": train_dataset_name,
                                "test_dataset": test_dataset_name, 
                                "isolator": isolator,
                                "energy_source": energy_source,
                                "feature_group": feature_group,
                                "model": model.name,
                                "model_path": model_path,
                                "energy_component": energy_component
                            }
                                
                            label_power_columns = [col for col in power_columns if energy_component in col]
                            # sum label value for all unit
                            # mean to squeeze value of power back
                            sum_power_label = predicted_data.groupby([TIMESTAMP_COL]).mean()[label_power_columns].sum(axis=1).sort_index()
                            # append predicted value to data_with_prediction
                            
                            # TO-DO: use predict_and_sort in train_isolator.py
                            
                            predicted_power_colname = get_predicted_power_colname(energy_component)
                            data_with_prediction[predicted_power_colname] = values
                            sum_predicted_power = data_with_prediction.groupby([TIMESTAMP_COL]).sum().sort_index()[predicted_power_colname]
                            if output_type == ModelOutputType.AbsPower:
                                item["mae"], item["mse"] = compute_error(sum_power_label, sum_predicted_power)
                            else:
                                # profile-based
                                min_watt, max_watt = get_min_max_watt(profiles, energy_component, node_type)
                                profile_watt = (min_watt + max_watt)/2
                                profile_reconstructed_power = sum_predicted_power + profile_watt
                                item["profile_mae"], item["profile_mse"]= compute_error(sum_power_label, profile_reconstructed_power)
                                item["profile_watt"] = profile_watt

                                # calculate background power cols (used by both abs-predicted and min)
                                predicted_background_power_values = predicted_background_power[energy_component]
                                background_power_colname = get_predicted_background_power_colname(energy_component)
                                background_data[background_power_colname] = predicted_background_power_values

                                predicted_background_dynamic_power_values = predicted_background_dynamic_power[energy_component]
                                dynamic_background_power_colname = get_predicted_dynamic_background_power_colname(energy_component)
                                background_data[dynamic_background_power_colname] = predicted_background_dynamic_power_values
                                
                                sorted_background_data = background_data.groupby([TIMESTAMP_COL]).sum().sort_index()
                                # abs-predicted based
                                sum_background_power = sorted_background_data[background_power_colname]
                                trained_reconstructed_power = sum_background_power + sum_predicted_power
                                item["train_bg_mae"], item["train_bg_mse"]= compute_error(sum_power_label, trained_reconstructed_power)
                                item["avg_train_bg"] = sum_background_power.mean()
                                item["bg_abs_model"] = abs_model.name
                                # min based
                                sum_dynamic_background_power = sorted_background_data[dynamic_background_power_colname]
                                min_reconstructed_power = sum_dynamic_background_power + min_watt + sum_predicted_power
                                item["min_bg_mae"], item["min_bg_mse"]= compute_error(sum_power_label, min_reconstructed_power)
                                item["min"] = min_watt       

if __name__ == '__main__':
    dataset_name = "sample_data"
    target_path = offline_trainer_output_path
    # same train/test dataset
    process(dataset_name, dataset_name, target_path)