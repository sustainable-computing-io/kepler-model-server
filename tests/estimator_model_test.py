# estimator_model_test.py
# - load_model
# - model.get_power()

import os
import sys
import pandas as pd


#################################################################
# import internal src 
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)
#################################################################

from estimate import load_model, default_predicted_col_func, compute_error
from train.trainer import model_toppath
from util.loader import get_model_group_path, DEFAULT_PIPELINE
from util import FeatureGroup, ModelOutputType, list_model_names
from util.prom_types import TIMESTAMP_COL

from isolator_test import test_isolators, get_isolate_results, isolator_output_path
from extractor_test import test_extractors, get_extract_results, test_energy_source, get_expected_power_columns, extractor_output_path

# extract_result, power_columns, corr, features = extractor.extract(query_results, energy_components, feature_group, energy_source, node_level) 
# test_data_with_label = extract_result // abs
# test_data_with_label = isolator.isolate(extract_result, _energy_source, label_cols=power_columns) // dyn
def test_model(group_path, model_name, test_data_with_label, power_columns, power_range = None):
    model_path = os.path.join(group_path, model_name)
    items = []
    model = load_model(model_path)
    predicted_power_map, data_with_prediction = model.append_prediction(test_data_with_label)
    for energy_component, _ in predicted_power_map.items():
        label_power_columns = [col for col in power_columns if energy_component in col]
        predicted_power_colname = default_predicted_col_func(energy_component)
        sum_power_label = test_data_with_label.groupby([TIMESTAMP_COL]).mean()[label_power_columns].sum(axis=1).sort_index()
        sum_predicted_power = data_with_prediction.groupby([TIMESTAMP_COL]).sum().sort_index()[predicted_power_colname]
        mae, mse, mape = compute_error(sum_power_label, sum_predicted_power)
        if power_range is None:
            power_range = sum_power_label.max() - sum_power_label.min()
        percent = mae/power_range
        item = dict()
        item['component'] = energy_component
        item['mae'] = mae
        item['mse'] = mse
        item['%mae'] = percent * 100
        item['mape'] = mape
        items += [item]
    return pd.DataFrame(items), data_with_prediction, model
        
# all energy_source, model_output, feature_group
def test_all_models(test_data_with_label, power_columns, output_type, feature_group, energy_source=test_energy_source):
    result_df_list = []
    group_path = get_model_group_path(model_toppath, output_type, feature_group, energy_source, assure=False, pipeline_name=DEFAULT_PIPELINE)
    model_names = list_model_names(group_path)
    for model_name in model_names:
        result_df, _, _ = test_model(group_path, model_name, test_data_with_label, power_columns)
        result_df['model_name'] = model_name
        result_df_list += [result_df]
    result_df = pd.concat(result_df_list)
    result_df['feature_group'] = feature_group.name
    result_df['output_type'] = output_type
    result_df['energy_source'] = energy_source
    return result_df

def process_all(extractors=test_extractors, isolators=test_isolators, isolate_save_path= isolator_output_path, extract_save_path=extractor_output_path):
    abs_test_list = []
    dyn_test_list = []

    for extractor in extractors:
        extractor_name = extractor.__class__.__name__
        extractor_results = get_extract_results(extractor_name, node_level=True, save_path=extract_save_path)
        for feature_group, result in extractor_results.items():
            print("Extractor ", extractor_name)
            power_columns=get_expected_power_columns()
            result_df = test_all_models(result, power_columns, ModelOutputType.AbsPower, FeatureGroup[feature_group])
            result_df['extractor'] = extractor_name
            abs_test_list += [result_df]

        for isolator in isolators:
            isolator_name = isolator.__class__.__name__
            isolator_results = get_isolate_results(isolator_name, extractor_name, save_path=isolate_save_path)
            for feature_group, result in isolator_results.items():
                print("Isolator ", isolator_name)
                result_df = test_all_models(result, power_columns, ModelOutputType.DynPower, FeatureGroup[feature_group])
                result_df['extractor'] = extractor_name
                result_df['isolator'] = isolator_name
                dyn_test_list += [result_df]

    abs_test_df = pd.concat(abs_test_list)
    dyn_test_df = pd.concat(dyn_test_list)
    return abs_test_df, dyn_test_df

if __name__ == '__main__':
    focus_columns = ['model_name', 'mae', 'mse']
    abs_train_df, dyn_train_df = process_all()
    print("Node-level test results:")
    print(abs_train_df.set_index(['energy_source', 'component', 'extractor'])[focus_columns].sort_values(by=['mae'], ascending=True))
    print("Container-level test results:")
    print(dyn_train_df.set_index(['energy_source', 'component', 'extractor', 'isolator'])[focus_columns].sort_values(by=['mae'], ascending=True))