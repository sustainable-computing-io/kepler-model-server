import os
import json
import joblib
import pandas as pd
from .saver import assure_path, METADATA_FILENAME, SCALER_FILENAME, WEIGHT_FILENAME, TRAIN_ARGS_FILENAME, NODE_TYPE_INDEX_FILENAME, MACHINE_SPEC_PATH, _pipeline_model_metadata_filename
from .train_types import ModelOutputType, FeatureGroup, PowerSourceMap, all_feature_groups
from urllib.request import urlopen

import requests
import codecs

major_version = "0.7"
version = "0.7.11"

FILTER_ITEM_DELIMIT = ";"
VALUE_DELIMIT = ":"
ARRAY_DELIMIT = ","

CHECKPOINT_FOLDERNAME = "checkpoint"
PREPROCESS_FOLDERNAME = "preprocessed_data"

##########################################################################
# pipeline loader

## default_train_output_pipeline: a default pipeline name which is output from the training pipeline
default_train_output_pipeline = "std_v{}".format(version)
default_pipelines = {
    "rapl-sysfs": "ec2-{}".format(version),
    "acpi": "specpower-{}".format(version)
}
base_model_url = "https://raw.githubusercontent.com/sustainable-computing-io/kepler-model-db/main/models/v{}".format(major_version)


def get_pipeline_url(model_topurl, pipeline_name):
    file_ext = ".zip"
    return os.path.join(model_topurl, pipeline_name + file_ext)


def assure_pipeline_name(pipeline_name, energy_source, nodeCollection):
    if pipeline_name == "":
        pipeline_name = default_pipelines[energy_source]
        if pipeline_name not in nodeCollection and default_train_output_pipeline in nodeCollection:
            pipeline_name = default_train_output_pipeline
    return pipeline_name


##########################################################################

default_trainer_name = "GradientBoostingRegressorTrainer"
default_node_type = 0
any_node_type = -1
default_feature_group = FeatureGroup.BPFOnly


def load_json(path, name):
    if ".json" not in name:
        name = name + ".json"
    filepath = os.path.join(path, name)
    try:
        with open(filepath) as f:
            res = json.load(f)
        return res
    except Exception:
        return None


def load_pkl(path, name):
    if ".pkl" not in name:
        name = name + ".pkl"
    filepath = os.path.join(path, name)
    try:
        res = joblib.load(filepath)
        return res
    except FileNotFoundError:
        return None
    except Exception as err:
        print("fail to load pkl {}: {}".format(filepath, err))
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

def load_remote_json(url_path):
    if ".json" not in url_path:
        url_path = url_path + ".json"
    try:        
        response = urlopen(url_path)
        response_data = response.read().decode('utf-8')
        json_data = json.loads(response_data)
        return json_data
    except:
        return None

def load_machine_spec(data_path, machine_id):
    machine_spec_path = os.path.join(data_path, MACHINE_SPEC_PATH)
    return load_json(machine_spec_path, machine_id)


def load_node_type_index(pipeline_path):
    return load_json(pipeline_path, NODE_TYPE_INDEX_FILENAME)


def load_metadata(model_path):
    return load_json(model_path, METADATA_FILENAME)


def load_train_args(pipeline_path):
    return load_json(pipeline_path, TRAIN_ARGS_FILENAME)


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
        data = data.apply(pd.to_numeric, errors="ignore")
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
        if key == "features":
            value = splits[1].split(ARRAY_DELIMIT)
        else:
            value = splits[1]
        filters[key] = value
    return filters


def is_valid_model(metadata, filters):
    for attrb, val in filters.items():
        if not hasattr(metadata, attrb) or getattr(metadata, attrb) is None:
            print("{} has no {}".format(metadata["model_name"], attrb))
            return False
        else:
            cmp_val = getattr(metadata, attrb)
            val = float(val)
            if attrb == "abs_max_corr":  # higher is better
                valid = cmp_val >= val
            else:  # lower is better
                valid = cmp_val <= val
            if not valid:
                return False
    return True


def get_model_name(trainer_name, node_type):
    return "{}_{}".format(trainer_name, node_type)


def get_node_type_from_name(model_name):
    return int(model_name.split("_")[-1])


def is_matched_type(nodeCollection, spec, pipeline_name, model_name, node_type, energy_source):
    model_node_type = get_node_type_from_name(model_name)
    if model_node_type == node_type:
        return True
    if node_type == any_node_type:
        # if not specify spec, return true
        if spec is None:
            return True
        # if covered by the model's node_type, return true
        pipeline_name = assure_pipeline_name(pipeline_name, energy_source, nodeCollection)
        if pipeline_name not in nodeCollection or nodeCollection[pipeline_name].node_type_index[model_node_type].cover(spec):
            return True
    return False


def get_largest_candidates(model_names, pipeline_name, nodeCollection, energy_source):
    pipeline_name = assure_pipeline_name(pipeline_name, energy_source, nodeCollection)
    if pipeline_name not in nodeCollection:
        return None
    node_type_index = nodeCollection[pipeline_name].node_type_index
    max_cores = 0
    candidates = dict()
    for model_name in model_names:
        model_node_type = get_node_type_from_name(model_name)
        if model_node_type not in node_type_index:
            continue
        cores = node_type_index[model_node_type].get_cores()
        if cores not in candidates:
            candidates[cores] = []
        candidates[cores] += [model_name]
        if cores > max_cores:
            max_cores = cores
    return candidates[max_cores] if max_cores > 0 else None


def get_pipeline_path(model_toppath, pipeline_name):
    return os.path.join(model_toppath, pipeline_name)


def get_model_group_path(model_toppath, output_type, feature_group, energy_source, pipeline_name, assure=True):
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
    return save_path + ".zip"


def download_and_save(url, filepath):
    try:
        response = requests.get(url)
    except Exception as e:
        print("Failed to load {} to {}: {}".format(url, filepath, e))
        return None
    if response.status_code != 200:
        print("Failed to load {} to {}: {}".format(url, filepath, response.status_code))
        return None
    with codecs.open(filepath, "wb") as f:
        f.write(response.content)
    print("Successfully load {} to {}".format(url, filepath))
    return filepath


def list_model_names(group_path):
    if not os.path.exists(group_path):
        return []
    model_names = [f.split(".")[0] for f in os.listdir(group_path) if ".zip" in f]
    return model_names


def list_pipelines(model_toppath, energy_source, model_type):
    pipeline_metadata_filename = _pipeline_model_metadata_filename(energy_source, model_type)
    pipeline_names = [f for f in os.listdir(model_toppath) if os.path.exists(os.path.join(model_toppath, f, pipeline_metadata_filename + ".csv"))]
    return pipeline_names


def list_all_abs_models(model_toppath, energy_source, valid_fgs, pipeline_name):
    abs_models_map = dict()
    for fg in valid_fgs:
        group_path = get_model_group_path(model_toppath, output_type=ModelOutputType.AbsPower, feature_group=fg, energy_source=energy_source, pipeline_name=pipeline_name, assure=False)
        model_names = list_model_names(group_path)
        abs_models_map[group_path] = model_names
    return abs_models_map


def list_all_dyn_models(model_toppath, energy_source, valid_fgs, pipeline_name):
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


def get_metadata_df(model_toppath, model_type, fg, energy_source, pipeline_name):
    group_path = get_model_group_path(model_toppath, output_type=ModelOutputType[model_type], feature_group=FeatureGroup[fg], energy_source=energy_source, pipeline_name=pipeline_name, assure=False)
    metadata_df = _get_metadata_df(group_path)
    if len(metadata_df) > 0:
        metadata_df[["trainer", "node_type"]] = metadata_df["model_name"].str.split("_", n=1, expand=True)
        metadata_df["node_type"] = metadata_df["node_type"].astype(int)
    return metadata_df, group_path


def get_all_metadata(model_toppath, pipeline_name, clean_empty=False):
    all_metadata = dict()
    for energy_source in PowerSourceMap.keys():
        all_metadata[energy_source] = dict()
        for model_type in list(ModelOutputType.__members__):
            all_fg_metadata = []
            for fg in all_feature_groups:
                metadata_df, group_path = get_metadata_df(model_toppath, model_type, fg, energy_source, pipeline_name)
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
    pipeline_metadata_filename = _pipeline_model_metadata_filename(energy_source, model_type)
    return load_csv(pipeline_path, pipeline_metadata_filename)


def get_download_output_path(download_path, energy_source, output_type) -> str:
    energy_source_path = assure_path(os.path.join(download_path, energy_source))
    return os.path.join(energy_source_path, output_type.name)


def get_url(output_type, feature_group, energy_source, trainer_name=default_trainer_name, node_type=default_node_type, model_topurl=base_model_url, pipeline_name=None, model_name=None, weight=False):
    if pipeline_name is None:
        pipeline_name = default_pipelines[energy_source]
    group_path = get_model_group_path(model_topurl, output_type=output_type, feature_group=feature_group, energy_source=energy_source, pipeline_name=pipeline_name, assure=False)
    if model_name is None:
        model_name = get_model_name(trainer_name, node_type)
    file_ext = ".zip"
    if weight:
        file_ext = ".json"
    return os.path.join(group_path, model_name + file_ext)


def class_to_json(class_obj):
    return json.loads(json.dumps(class_obj.__dict__))


def get_version_path(output_path, assure=True):
    version_path = os.path.join(output_path, "v{}".format(major_version))
    if assure:
        return assure_path(version_path)
    return version_path


def get_export_path(output_path, pipeline_name, assure=True):
    version_path = get_version_path(output_path)
    export_path = os.path.join(version_path, pipeline_name)
    if assure:
        return assure_path(export_path)
    return export_path


def get_preprocess_folder(pipeline_path, assure=True):
    preprocess_folder = os.path.join(pipeline_path, PREPROCESS_FOLDERNAME)
    if assure:
        return assure_path(preprocess_folder)
    return preprocess_folder


def get_general_filename(prefix, energy_source, fg, ot, extractor, isolator=None):
    fg_suffix = "" if fg is None else "_" + fg.name
    if ot.name == ModelOutputType.DynPower.name:
        return "{}_dyn_{}_{}_{}{}".format(prefix, extractor, isolator, energy_source, fg_suffix)
    if ot.name == ModelOutputType.AbsPower.name:
        return "{}_abs_{}_{}{}".format(prefix, extractor, energy_source, fg_suffix)
    return None
