import os
import json
import joblib
import pandas as pd
from saver import assure_path, METADATA_FILENAME, SCALER_FILENAME, WEIGHT_FILENAME, _pipeline_metadata_filename
from train_types import ModelOutputType, FeatureGroup, PowerSourceMap, all_feature_groups
from urllib.request import urlopen

import requests
import codecs

FILTER_ITEM_DELIMIT = ';'
VALUE_DELIMIT = ':'
ARRAY_DELIMIT = ','

DEFAULT_PIPELINE = 'default'
CHECKPOINT_FOLDERNAME = 'checkpoint'
DOWNLOAD_FOLDERNAME = 'download'

default_init_model_url = "https://github.com/sunya-ch/kepler-model-db/raw/main/models/"
default_init_pipeline_name = "trl-nx12_mixed_all_TrainIsolator"
default_trainer_name = "GradientBoostingRegressorTrainer"
default_node_type = "1"
default_feature_group = FeatureGroup.CgroupOnly

def load_json(path, name):
    if ".json" not in name:
        name = name + ".json"
    filepath = os.path.join(path, name)
    try:
        with open(filepath) as f:
            res = json.load(f)
        return res
    except:
        return None
    
def load_pkl(path, name):
    if ".pkl" not in name:
        name = name + ".pkl"
    filepath = os.path.join(path, name)
    try:
        res = joblib.load(filepath)
        return res
    except Exception:
        return None
   
def load_remote_pkl(url_path):
    if ".pkl" not in url_path:
        url_path = url_path + ".pkl"
    try:        
        response = urlopen(url_path)
        loaded_model = joblib.load(response)
        return loaded_model
    except:
        return None

def load_metadata(model_path):
    return load_json(model_path, METADATA_FILENAME)

def load_scaler(model_path):
    return load_pkl(model_path, SCALER_FILENAME)

def load_weight(model_path):
    return load_json(model_path, WEIGHT_FILENAME)

def load_profile(profile_path, source):
    profile_filename = os.path.join(profile_path, source + ".json")
    if not os.path.exists(profile_filename):
        profile = dict()
        for component in PowerSourceMap[source]:
            profile[component] = dict()
    else:
        with open(profile_filename) as f:
            profile = json.load(f)
    return profile

def load_csv(path, name):
    csv_file = name + ".csv"
    file_path = os.path.join(path, csv_file)
    try:
        data = pd.read_csv(file_path)
        data = data.apply(pd.to_numeric, errors='ignore')
        return data
    except:
        # print('cannot load {}'.format(file_path))
        return None

def parse_filters(filter):
    filter_list = filter.split(FILTER_ITEM_DELIMIT)
    filters = dict()
    for filter_item in filter_list:
        splits = filter_item.split(VALUE_DELIMIT)
        if len(splits) != 2:
            continue
        key = splits[0]
        if key == 'features':
            value = splits[1].split(ARRAY_DELIMIT)
        else:
            value = splits[1]
        filters[key] = value
    return filters 

def is_valid_model(metadata, filters):
    for attrb, val in filters.items():
        if not hasattr(metadata, attrb) or getattr(metadata, attrb) is None:
            print('{} has no {}'.format(metadata['model_name'], attrb))
            return False
        else:
            cmp_val = getattr(metadata, attrb)
            val = float(val)
            if attrb == 'abs_max_corr': # higher is better
                valid = cmp_val >= val
            else: # lower is better
                valid = cmp_val <= val
            if not valid:
                return False
    return True

def get_model_name(trainer_name, node_type):
    return "{}_{}".format(trainer_name, node_type)

def get_pipeline_path(model_toppath, pipeline_name=DEFAULT_PIPELINE):
    return os.path.join(model_toppath, pipeline_name)

def get_model_group_path(model_toppath, output_type, feature_group, energy_source='rapl', pipeline_name=DEFAULT_PIPELINE, assure=True):
    pipeline_path = get_pipeline_path(model_toppath, pipeline_name)
    energy_source_path = os.path.join(pipeline_path, energy_source)
    output_path = os.path.join(energy_source_path, output_type.name)
    feature_path = os.path.join(output_path, feature_group.name)
    if assure:
        assure_path(pipeline_path)
        assure_path(energy_source_path)
        assure_path(output_path)
        assure_path(feature_path)
    return feature_path

def get_save_path(group_path, trainer_name, node_type):
    return os.path.join(group_path, get_model_name(trainer_name, node_type))

def get_archived_file(group_path, model_name):
    save_path = os.path.join(group_path, model_name)
    return save_path + '.zip'

def download_and_save(url, filepath):
    try:
        response = requests.get(url)
    except Exception as e:
        print("Failed to load {} to {}: {}".format(url, filepath, e))
        return None
    if response.status_code != 200:
        print("Failed to load {} to {}: {}".format(url, filepath, response.status_code))
        return None
    with codecs.open(filepath, 'wb') as f:
        f.write(response.content)
    print("Successfully load {} to {}".format(url, filepath))
    return filepath

def list_model_names(model_path):
    model_names = [f.split('.')[0] for f in os.listdir(model_path) if '.zip' in f]
    return model_names

def list_all_abs_models(model_toppath, energy_source, valid_fgs, pipeline_name=DEFAULT_PIPELINE):
    abs_models_map = dict()
    for fg in valid_fgs:
        group_path = get_model_group_path(model_toppath, output_type=ModelOutputType.AbsPower, feature_group=fg, energy_source=energy_source, pipeline_name=pipeline_name, assure=False)
        model_names = list_model_names(group_path)
        abs_models_map[group_path] = model_names
    return abs_models_map

def list_all_dyn_models(model_toppath, energy_source, valid_fgs, pipeline_name=DEFAULT_PIPELINE):
    dyn_models_map = dict()
    for fg in valid_fgs:
        group_path = get_model_group_path(model_toppath, output_type=ModelOutputType.DynPower, feature_group=fg, energy_source=energy_source, pipeline_name=pipeline_name, assure=False)
        model_names = list_model_names(group_path)
        dyn_models_map[group_path] = model_names
    return dyn_models_map

def _get_metadata_df(group_path):
    metadata_items = []
    if os.path.exists(group_path):
        model_names = list_model_names(group_path)
        for model_name in model_names:
            model_path = os.path.join(group_path, model_name)
            metadata = load_metadata(model_path)
            if metadata is not None:
                metadata_items += [metadata]
    return pd.DataFrame(metadata_items)

def get_all_metadata(model_toppath, pipeline_name, clean_empty=False):
    all_metadata = dict()
    for energy_source in PowerSourceMap.keys():
        all_metadata[energy_source] = dict()
        for model_type in list(ModelOutputType.__members__):
            all_fg_metadata = []
            for fg in all_feature_groups:
                group_path = get_model_group_path(model_toppath, output_type=ModelOutputType[model_type], feature_group=FeatureGroup[fg], energy_source=energy_source, pipeline_name=pipeline_name, assure=False)
                metadata_df = _get_metadata_df(group_path)
                if len(metadata_df) > 0:
                    metadata_df["feature_group"] = fg
                    all_fg_metadata += [metadata_df]
                elif clean_empty:
                    os.rmdir(group_path)
            if len(all_fg_metadata) > 0:
                all_metadata[energy_source][model_type] = pd.concat(all_fg_metadata)
            elif clean_empty:
                energy_source_path = os.path.join(model_toppath, energy_source)
                os.rmdir(energy_source_path)

    return all_metadata
       
def load_pipeline_metadata(pipeline_path, energy_source, model_type):
    pipeline_metadata_filename = _pipeline_metadata_filename(energy_source, model_type)
    return load_csv(pipeline_path, pipeline_metadata_filename)

def get_download_path():
    return os.path.join(os.path.dirname(__file__), DOWNLOAD_FOLDERNAME)

def get_download_output_path(output_type):
    return os.path.join(get_download_path(), output_type.name)

def get_url(output_type, feature_group=default_feature_group, trainer_name=default_trainer_name, node_type=default_node_type, model_topurl=default_init_model_url, energy_source="rapl", pipeline_name=default_init_pipeline_name):
    group_path = get_model_group_path(model_topurl, output_type=output_type, feature_group=feature_group, energy_source=energy_source, pipeline_name=pipeline_name, assure=False)
    model_name = get_model_name(trainer_name, node_type)
    return os.path.join(group_path, model_name + ".zip")

def class_to_json(class_obj):
    return json.loads(json.dumps(class_obj.__dict__))