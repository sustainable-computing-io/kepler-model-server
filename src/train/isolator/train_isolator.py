import os
import sys
import numpy as np

util_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'util')
sys.path.append(util_path)
estimate_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'estimate')
sys.path.append(estimate_path)

from isolator import Isolator, isolate_container
from estimate import load_model, get_predicted_power_colname, get_predicted_background_power_colname, get_dynamic_power_colname, get_reconstructed_power_colname, get_label_power_colname, get_background_containers
from extractor import find_correlations
from preprocess import get_extracted_power_labels

from util import PowerSourceMap
from util.train_types import get_valid_feature_groups
from util.prom_types import TIMESTAMP_COL
from util.extract_types import container_level_index, container_id_colname, col_to_component
from util.config import model_toppath
from util.loader import list_all_abs_models, DEFAULT_PIPELINE


def is_better(curr_min_err, err, curr_max_corr, corr, corr_threshold=0.7):
    if curr_min_err is None:
        return True
    if corr >= corr_threshold:
        if err == curr_min_err:
            # decide by corr if equal err
            return corr > curr_max_corr 
        return err < curr_min_err
    elif curr_max_corr < corr_threshold and corr >= curr_max_corr:
        # when better accuracy and better corr but less than threshold
        return err < curr_min_err
    return False

def get_abs_models(workload_feature_cols, energy_source, toppath=model_toppath, pipeline_name=DEFAULT_PIPELINE):
    # from abs_model_path
    # find valid_feature_groups
    # list_model_names
    # load each model 
    abs_models = []
    valid_fgs = get_valid_feature_groups(workload_feature_cols)
    abs_models_map = list_all_abs_models(toppath, energy_source, valid_fgs, pipeline_name=pipeline_name)
    for group_path, model_names in abs_models_map.items():
        for model_name in model_names:
            model_path = os.path.join(group_path, model_name)
            model = load_model(model_path)
            abs_models += [model]
    return abs_models

# extracted_power_labels: sum of power labels over sorted timestamp for each energy_components
# background_powers: sum of predicted background powers over sorted timestamp for each energy_components
def append_dyn_power(target_data, energy_components, extracted_power_labels, background_powers, min_val=0):
    appended_data = target_data.set_index([TIMESTAMP_COL])
    power_join_df = extracted_power_labels.join(background_powers).fillna(0)
    for energy_component in energy_components:
        dyn_colname = get_dynamic_power_colname(energy_component)
        bg_colname = get_predicted_background_power_colname(energy_component)
        label_colname = get_label_power_colname(energy_component)
        power_join_df[dyn_colname] = power_join_df[label_colname] - power_join_df[bg_colname]
        appended_data = appended_data.join(power_join_df[[dyn_colname]])
    num = appended_data._get_numeric_data().astype(float)
    num[num < min_val] = min_val
    return appended_data

def get_target_data_with_dyn_power(model, energy_components, extracted_power_labels, target_data, background_data):
    # predict background power from the rest usage (background container usage)
    sum_background_data = background_data.groupby([TIMESTAMP_COL]).sum()
    _, sum_background_data_with_prediction = model.append_prediction(sum_background_data, predicted_col_func=get_predicted_background_power_colname)
    # sum over predicted value
    background_power_colnames = [get_predicted_background_power_colname(energy_component) for energy_component in energy_components]
    background_powers = sum_background_data_with_prediction[background_power_colnames]
    # append_dyn_power
    target_data_with_dyn_power = append_dyn_power(target_data, energy_components, extracted_power_labels, background_powers)

    return target_data_with_dyn_power, sum_background_data_with_prediction

# traverse all abs_model with minimum mae for each energy_source
def find_best_target_data_with_dyn_power(energy_source, energy_components, extracted_data, background_containers, label_cols, toppath=model_toppath, pipeline_name=DEFAULT_PIPELINE):
    workload_feature_cols = [col for col in extracted_data.columns if col not in label_cols and col not in container_level_index and 'ratio' not in col and 'node' not in col]
    curr_min_err = None
    curr_max_corr= None
    best_target_data_with_dyn_power = None
    best_background_data_with_prediction  = None
    abs_models = get_abs_models(workload_feature_cols, energy_source, toppath=toppath, pipeline_name=pipeline_name)
    # get background power
        # isolate target container usage
    target_data, background_data = isolate_container(extracted_data, background_containers, label_cols)
    # get_extracted_power_labels
    extracted_power_labels = get_extracted_power_labels(extracted_data, energy_components, label_cols)
    for model in abs_models:
       err = model.mae
       # dynamic power = remove background power from label power
       target_data_with_dyn_power, background_data_with_prediction = get_target_data_with_dyn_power(model, energy_components, extracted_power_labels, target_data, background_data)
       # find correlation between dynamic power and target container usage
       dyn_power_cols = [get_dynamic_power_colname(energy_component) for energy_component in energy_components]
       corr_data = find_correlations(energy_source, target_data_with_dyn_power.groupby([TIMESTAMP_COL]).sum(), dyn_power_cols, workload_feature_cols).dropna()
       try:
           corr = corr_data.values.max()
           if np.isnan(corr):
               corr = 0
       except:
           corr = 0
       # is_better
       if is_better(curr_min_err, err, curr_max_corr, corr):
           curr_min_err = err
           curr_max_corr = corr
           best_target_data_with_dyn_power = target_data_with_dyn_power
           best_background_data_with_prediction = background_data_with_prediction
    return best_target_data_with_dyn_power, best_background_data_with_prediction

# TO-DO: suppport multiple node types
class TrainIsolator(Isolator):
    def __init__(self, idle_data, profiler, abs_pipeline_name=DEFAULT_PIPELINE):
        self.idle_data = idle_data
        self.profiles = profiler.process(self.idle_data)
        self.background_containers = get_background_containers(self.idle_data)
        self.abs_pipeline_name = abs_pipeline_name

    def isolate(self, data, label_cols, energy_source):
        index_list = data.index.names
        if index_list[0] is not None:
            data = data.reset_index()
        energy_components = PowerSourceMap[energy_source]
        label_cols = list(label_cols)
        best_target_data_with_dyn_power, _ = find_best_target_data_with_dyn_power(energy_source, energy_components, data, self.background_containers, label_cols, pipeline_name=self.abs_pipeline_name)
        isolated_data = best_target_data_with_dyn_power.copy()
        power_label_cols = [get_label_power_colname(energy_component) for energy_component in energy_components]
        extracted_power_labels = get_extracted_power_labels(data, energy_components, label_cols)[power_label_cols]
        isolated_data = isolated_data.join(extracted_power_labels)
        for label_col in label_cols:
            energy_component = col_to_component(label_col)
            component_label_colname = get_label_power_colname(energy_component)
            dyn_power_colname = get_dynamic_power_colname(energy_component)
            isolated_data[label_col] = isolated_data[label_col]/isolated_data[component_label_colname]*isolated_data[dyn_power_colname]
        drop_cols = power_label_cols + [get_dynamic_power_colname(energy_component) for energy_component in energy_components]
        isolated_data = isolated_data.drop(columns=drop_cols).reset_index().fillna(0)
        if index_list[0] is not None:
            isolated_data = isolated_data.set_index(index_list)
        return isolated_data
    
    def reconstruct(self, extracted_data, data_with_prediction, energy_source, label_cols):
        reconstructed_data = data_with_prediction.groupby([TIMESTAMP_COL]).sum()
        energy_components = PowerSourceMap[energy_source]
        _, background_data_with_prediction = find_best_target_data_with_dyn_power(energy_source, energy_components, extracted_data, self.background_containers, label_cols)
        background_power_colnames = [get_predicted_background_power_colname(energy_component) for energy_component in energy_components]
        background_powers = background_data_with_prediction[background_power_colnames].groupby([TIMESTAMP_COL]).sum()
        reconstructed_data = reconstructed_data.join(background_powers)
        for energy_component in energy_components:
            predicted_colname = get_predicted_power_colname[energy_source]
            background_power_colname = get_predicted_background_power_colname(energy_component)
            reconstructed_data[get_reconstructed_power_colname(energy_component)] = data_with_prediction[predicted_colname]  + background_data_with_prediction[background_power_colname]
        return reconstructed_data