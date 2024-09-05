"""
python src/train/specpower_pipeline.py

This program trains a pipeline called `specpower` from the preprocssed data from SPECPower database.

required step:
- run the following command to serve kepler_spec_power_db
    docker run -it -p 8080:80 quay.io/sustainability/kepler_spec_power_db:v0.7
"""

import datetime
import json
import os
from io import StringIO

import pandas as pd
import requests

from kepler_model.train.extractor import DefaultExtractor
from kepler_model.train.isolator.isolator import MinIdleIsolator
from kepler_model.train.pipeline import NewPipeline
from kepler_model.train.profiler.node_type_index import NodeTypeSpec
from kepler_model.util.extract_types import component_to_col
from kepler_model.util.format import time_to_str
from kepler_model.util.prom_types import TIMESTAMP_COL, node_info_column
from kepler_model.util.train_types import BPF_FEATURES, FeatureGroup, PowerSourceMap, default_trainer_names

platform_energy_source = "acpi"
acpi_component = PowerSourceMap[platform_energy_source][0]
acpi_label = component_to_col(acpi_component)


def read_csv_from_url(topurl, path):
    response = requests.get(os.path.join(topurl, path))
    if response.status_code == 200:
        try:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            return df
        except Exception as e:
            print("cannot read csv from url: ", e)
            return None
    print("cannot read csv from url: ", response.status_code)
    return None


def read_json_from_url(topurl, path):
    response = requests.get(os.path.join(topurl, path))
    if response.status_code == 200:
        try:
            json_data = json.loads(response.text)
            return json_data
        except Exception as e:
            print("cannot read json from url: ", e)
            return None
    print("cannot read json from url: ", response.status_code)
    return None


def get_machine_spec(df):
    if len(df) == 0:
        return NodeTypeSpec()
    data_spec = df.iloc[0].to_dict()
    data_spec["memory"] = data_spec["memory_gb"]
    data_spec["frequency"] = data_spec["cpu_freq_mhz"]
    spec = NodeTypeSpec(**data_spec)
    return spec


class SpecPipelineRun:
    def __init__(self, name, abs_trainer_names=default_trainer_names, dyn_trainer_names=default_trainer_names, isolator=MinIdleIsolator()):
        self.feature_group = FeatureGroup.BPFOnly
        self.power_labels = [acpi_label]
        self.energy_source = platform_energy_source
        # extractor is not used
        self.pipeline = NewPipeline(
            name,
            abs_trainer_names=abs_trainer_names,
            dyn_trainer_names=dyn_trainer_names,
            extractor=DefaultExtractor(),
            isolator=isolator,
            target_energy_sources=[self.energy_source],
            valid_feature_groups=[self.feature_group],
        )

    def load_spec_data(self, spec_db_url):
        spec_extracted_data = dict()
        # load index.json
        file_indexes = read_json_from_url(spec_db_url, "index.json")
        if file_indexes is not None:
            for filename in file_indexes:
                machine_id, _ = os.path.splitext(filename)
                df = read_csv_from_url(spec_db_url, filename)
                if df is not None:
                    # find node_type
                    spec = get_machine_spec(df)
                    node_type = self.pipeline.node_collection.index_train_machine(machine_id, spec)
                    df[node_info_column] = node_type
                    # select only needed column
                    spec_extracted_data[machine_id] = df[[TIMESTAMP_COL, node_info_column, acpi_label] + BPF_FEATURES]
            self.pipeline.node_collection.save()
        return spec_extracted_data

    def process(self, spec_db_url):
        spec_extracted_data = self.load_spec_data(spec_db_url)
        abs_data = pd.concat(spec_extracted_data.values(), ignore_index=True)
        df_list = []
        for df in spec_extracted_data.values():
            isolated_df = self.pipeline.isolator.isolate(df, label_cols=self.power_labels, energy_source=self.energy_source)
            df_list += [isolated_df]
        dyn_data = pd.concat(df_list, ignore_index=True)
        self.pipeline._train(abs_data, dyn_data, self.power_labels, self.energy_source, self.feature_group.name)
        self.pipeline.print_pipeline_process_end(self.energy_source, self.feature_group.name, abs_data, dyn_data)
        self.pipeline.metadata["last_update_time"] = time_to_str(datetime.datetime.utcnow())
        return True, abs_data, dyn_data

    def save_metadata(self):
        self.pipeline.save_metadata()

    def archive_pipeline(self):
        self.pipeline.archive_pipeline()


if __name__ == "__main__":
    spec_db_url = "http://localhost:8080"
    pipeline_name = "specpower-0.7.11"
    pipelinerun = SpecPipelineRun(name=pipeline_name)
    _, abs_data, dyn_data = pipelinerun.process(spec_db_url)
    pipelinerun.save_metadata()
    pipelinerun.archive_pipeline()
