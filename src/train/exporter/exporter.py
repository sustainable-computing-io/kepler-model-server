import os
import sys

import datetime
import shutil
import pandas as pd

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

cur_path = os.path.join(os.path.dirname(__file__))
sys.path.append(cur_path)

from validator import find_acceptable_mae
from train_types import ModelOutputType, PowerSourceMap, FeatureGroup
from loader import load_csv, load_pipeline_metadata, get_model_group_path, load_metadata, load_train_args, get_preprocess_folder, get_general_filename
from saver import WEIGHT_FILENAME, save_pipeline_metadata, save_train_args
from format import time_to_str
from writer import generate_pipeline_page, generate_validation_results, append_version_readme

def export(data_path, pipeline_path, machine_path, machine_id, version, publisher, collect_date, include_raw=False):
    pipeline_metadata = load_metadata(pipeline_path)
    if pipeline_metadata is None:
        print("no pipeline metadata")
        return
    
    pipeline_name = pipeline_metadata["name"]
    # update pipeline metadata
    pipeline_metadata["version"] = version
    pipeline_metadata["publisher"] = publisher
    pipeline_metadata["collect_time"] = time_to_str(collect_date)
    pipeline_metadata["export_time"] = time_to_str(datetime.datetime.utcnow())

    input_path = os.path.join(pipeline_path, "..")
    out_pipeline_path = os.path.join(machine_path, pipeline_name)

    preprocess_folder = get_preprocess_folder(pipeline_path, assure=False)
    if include_raw:
        # copy preprocessed_data
        if os.path.exists(preprocess_folder):
            destination_folder = get_preprocess_folder(machine_path)
            shutil.copytree(preprocess_folder, destination_folder, dirs_exist_ok=True)
        else:
            print("cannot export raw data: {} does not exist.", preprocess_folder)
        
    extractor = pipeline_metadata["extractor"]
    isolator = pipeline_metadata["isolator"]
    mae_validated_df_map = dict()
    for energy_source in PowerSourceMap.keys():
        mae_validated_df_map[energy_source] = dict()
        for ot in ModelOutputType:
            metadata_df = load_pipeline_metadata(pipeline_path, energy_source, ot.name)
            if metadata_df is None:
                print("no metadata for ", energy_source, ot.name)
                continue
            mae_validated_list = [] 
            for _, row in metadata_df.iterrows():
                model_name = row["model_name"]
                fg = FeatureGroup[row["feature_group"]]
                preprocess_filename = get_general_filename("preprocess", energy_source, fg, ot, extractor, isolator)
                preprocess_data = load_csv(preprocess_folder, preprocess_filename)
                is_acceptable, power_range, mae_threshold = find_acceptable_mae(preprocess_data, row)
                if is_acceptable:
                    mae_validated_list += [row.to_dict()]
                    out_group_path = get_model_group_path(machine_path, ot, fg, energy_source, assure=True, pipeline_name=pipeline_name)
                    in_group_path = get_model_group_path(input_path, ot, fg, energy_source, assure=False, pipeline_name=pipeline_name)
                    out_model_path = os.path.join(out_group_path, model_name)
                    in_model_path = os.path.join(in_group_path, model_name)
                    # copy zip file
                    shutil.copy(in_model_path + ".zip", out_model_path + ".zip")
                    # copy weight if exists
                    weight_filename = WEIGHT_FILENAME + ".json"
                    if weight_filename in os.listdir(in_model_path):
                        shutil.copy(os.path.join(in_model_path, weight_filename), out_model_path + ".json")
            mae_validated_df = pd.DataFrame(mae_validated_list)
            if len(mae_validated_df) > 0:
                mae_validated_df["power_range"] = power_range
                mae_validated_df["mae_threshold"] = mae_threshold
                # save pipeline metadata 
                save_pipeline_metadata(out_pipeline_path, pipeline_metadata, energy_source, ot.name, mae_validated_df)
                print("Exported models for {}/{}".format(energy_source, ot.name))
                print(mae_validated_df)
                mae_validated_df_map[energy_source][ot.name] = mae_validated_df
            else:
                print("No valid models exported for {}/{}".format(energy_source, ot.name))

    train_args = load_train_args(pipeline_path)
    train_args["machine_id"] = machine_id

    # save train args
    save_train_args(out_pipeline_path, train_args)

    # generate document
    generate_pipeline_page(data_path, machine_path, train_args)
    generate_validation_results(machine_path, train_args, mae_validated_df_map)
    append_version_readme(machine_path, train_args, pipeline_metadata, include_raw)


