import os
import sys
import pandas as pd
import shutil

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

cur_path = os.path.join(os.path.dirname(__file__))
sys.path.append(cur_path)

from train_types import ModelOutputType, PowerSourceMap, FeatureGroup, weight_support_trainers
from loader import load_pipeline_metadata, get_model_group_path, load_weight, get_archived_file
from saver import save_json

mae_threshold = 10
mape_threshold = 20

class ExportModel():
    def __init__(self, models_path, output_type, feature_group, energy_source, pipeline_name, model_name, metadata):
        self.pipeline_name = pipeline_name
        self.energy_source = energy_source
        self.output_type = output_type
        self.feature_group = feature_group
        self.model_name = model_name
        self.source_model_group_path = get_model_group_path(models_path, output_type, feature_group, energy_source, assure=False, pipeline_name=pipeline_name)
        self.source_model_path = os.path.join(self.source_model_group_path, self.model_name)
        self.source_model_zip = get_archived_file(self.source_model_group_path, self.model_name)
        self.metadata = metadata
        self.node_type = metadata["node_type"]
        trainer = metadata["trainer"]
        self.weight = None
        if trainer in weight_support_trainers:
            self.weight = load_weight(self.source_model_path)

    # assure if version_path is local
    def get_export_path(self, version_path, assure):
        target_model_path =  get_model_group_path(version_path, self.output_type, self.feature_group, self.energy_source, assure=assure, pipeline_name=self.pipeline_name)
        return target_model_path

    def export(self, local_version_path):
        target_model_group_path = self.get_export_path(local_version_path, assure=True)
        if self.weight is not None:
            save_json(path=target_model_group_path, name=self.model_name, data=self.weight)   
        target_model_zip = get_archived_file(target_model_group_path, self.model_name)
        shutil.copy(self.source_model_zip, target_model_zip)

    def get_weight_filepath(self, version_path):
        return os.path.join(self.get_export_path(version_path, assure=False), self.model_name + ".json")
    
    def get_archived_filepath(self, version_path):
        return get_archived_file(self.get_export_path(version_path, assure=False), self.model_name)


class BestModelCollection():
    def __init__(self, error_key):
        self.error_key = error_key
        self.collection = dict()
        self.weight_collection = dict()
        for energy_source in PowerSourceMap.keys():
            self.collection[energy_source] = dict()
            self.weight_collection[energy_source] = dict()
            for ot in ModelOutputType:
                self.collection[energy_source][ot.name] = dict()
                self.weight_collection[energy_source][ot.name] = dict()
                for fg in FeatureGroup:
                    self.collection[energy_source][ot.name][fg.name] = None
                    self.weight_collection[energy_source][ot.name][fg.name] = None
        self.has_model = False

    def compare_new_item(self, export_item):
        current_best = self.collection[export_item.energy_source][export_item.output_type.name][export_item.feature_group.name] 
        compare_mape = export_item.metadata[self.error_key]
        if current_best is None or current_best.metadata[self.error_key] > compare_mape:
            self.collection[export_item.energy_source][export_item.output_type.name][export_item.feature_group.name] = export_item
            self.has_model = True
        if export_item.weight is not None:
            current_best = self.weight_collection[export_item.energy_source][export_item.output_type.name][export_item.feature_group.name] 
            if current_best is None or current_best.metadata[self.error_key] > compare_mape:
                self.weight_collection[export_item.energy_source][export_item.output_type.name][export_item.feature_group.name] = export_item
        
    def get_best_item(self, energy_source, output_type_name, feature_group_name):
        return self.collection[energy_source][output_type_name][feature_group_name]
    
    def get_best_item_with_weight(self, energy_source, output_type_name, feature_group_name):
        return self.weight_collection[energy_source][output_type_name][feature_group_name]

# get_validated_export_items return valid export items
def get_validated_export_items(pipeline_path, pipeline_name):
    export_items = []
    valid_metadata_df = dict()
    models_path = os.path.join(pipeline_path, "..")
    for energy_source in PowerSourceMap.keys():
        valid_metadata_df[energy_source] = dict()
        for ot in ModelOutputType:
            metadata_df = load_pipeline_metadata(pipeline_path, energy_source, ot.name)
            if metadata_df is None:
                print("no metadata for", energy_source, ot.name)
                continue
            valid_rows = []
            for _, row in metadata_df.iterrows():
                if row['mape'] <= mape_threshold or row['mae'] <= mae_threshold:
                    model_name = row["model_name"]
                    fg = FeatureGroup[row["feature_group"]]
                    export_item = ExportModel(models_path, ot, fg, energy_source, pipeline_name, model_name, row.to_dict())
                    source_file = export_item.source_model_zip
                    if not os.path.exists(source_file):
                        print("source not exist: ", source_file)
                        continue
                    export_items += [export_item]
                    valid_rows += [row]
            valid_metadata_df[energy_source][ot.name] = pd.DataFrame(valid_rows)
    return export_items, valid_metadata_df