import json
import joblib
import os

METADATA_FILENAME = 'metadata'
SCALER_FILENAME = 'scaler'
WEIGHT_FILENAME = 'weight'
TRAIN_ARGS_FILENAME = 'train_arguments'

def _pipeline_model_metadata_filename(energy_source, model_type):
    return "{}_{}_model_metadata".format(energy_source, model_type)

def assure_path(path):
    if path == '':
        return ''
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def save_json(path, name, data):
    if '.json' not in name:
        name = name + '.json'
    assure_path(path)
    filename = os.path.join(path, name)   
    with open(filename, "w") as f:
        json.dump(data, f)
    return name

def save_pkl(path, name, data):
    if '.pkl' not in name:
        name = name + '.pkl'
    assure_path(path)
    filename = os.path.join(path, name)   
    joblib.dump(data, filename)
    return name

def save_csv(path, name, data):
    if '.csv' not in name:
        name = name + '.csv'
    assure_path(path)
    filename = os.path.join(path, name)   
    data.to_csv(filename)
    return name

def save_metadata(model_path, metadata):
    return save_json(model_path, METADATA_FILENAME, metadata)

def save_train_args(pipeline_path, args):
    return save_json(pipeline_path, TRAIN_ARGS_FILENAME, args)

def save_scaler(model_path, scaler):
    return save_pkl(model_path, SCALER_FILENAME, scaler)

def save_weight(model_path, weight):
    return save_json(model_path, WEIGHT_FILENAME, weight)

def save_pipeline_metadata(pipeline_path, pipeline_metadata, energy_source, model_type, metadata_df):
    save_metadata(pipeline_path, pipeline_metadata)
    pipeline_model_metadata_filename = _pipeline_model_metadata_filename(energy_source, model_type)
    return save_csv(pipeline_path, pipeline_model_metadata_filename, metadata_df)

def save_profile(profile_path, source, profile):
    profile_filename = os.path.join(profile_path, source + ".json")
    with open(profile_filename, "w") as f:
        json.dump(profile, f)