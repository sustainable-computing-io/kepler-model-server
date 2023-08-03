import os
import sys
import threading
import pandas as pd

util_path = os.path.join(os.path.dirname(__file__), '..', 'util')
sys.path.append(util_path)
extractor_path = os.path.join(os.path.dirname(__file__), 'extractor')
sys.path.append(extractor_path)
isolator_path = os.path.join(os.path.dirname(__file__), 'isolator')
sys.path.append(isolator_path)

from extractor import DefaultExtractor
from isolator import MinIdleIsolator

from train_types import PowerSourceMap, FeatureGroups
from config import model_toppath, ERROR_KEY
from loader import get_all_metadata, get_pipeline_path
from saver import save_pipeline_metadata

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

def load_class(module_name, class_name):
    path = os.path.join(os.path.dirname(__file__), '{}/{}'.format(module_name, class_name))
    sys.path.append(path)
    import importlib
    module_path = importlib.import_module('train.{}.{}.main'.format(module_name, class_name))
    return getattr(module_path, class_name)

def run_train(trainer, data, power_labels, pipeline_lock):
    trainer.process(data, power_labels, pipeline_lock=pipeline_lock)

class Pipeline():
    def __init__(self, name, trainers, extractor, isolator):
        self.extractor = extractor
        self.isolator = isolator
        self.trainers = trainers
        self.name = name
        self.lock = threading.Lock()
        self.path = get_pipeline_path(model_toppath=model_toppath ,pipeline_name=self.name)

    def get_abs_data(self, query_results, energy_components, feature_group, energy_source, aggr):
        extracted_data, power_labels, _ = self.extractor.extract(query_results, energy_components, feature_group, energy_source, node_level=True, aggr=aggr)
        return extracted_data, power_labels
    
    def get_dyn_data(self, query_results, energy_components, feature_group, energy_source):
        extracted_data, power_labels, _ = self.extractor.extract(query_results, energy_components, feature_group, energy_source, node_level=False)
        isolated_data = self.isolator.isolate(extracted_data, label_cols=power_labels, energy_source=energy_source)
        return isolated_data

    def _train(self, abs_data, dyn_data, power_labels, energy_source, feature_group):
        # start the thread pool
        with ThreadPoolExecutor(2) as executor:
            futures = []
            for trainer in self.trainers:
                if trainer.feature_group_name != feature_group:
                    continue
                if trainer.energy_source != energy_source:
                    continue
                if trainer.node_level:
                    future = executor.submit(run_train, trainer, abs_data, power_labels, pipeline_lock=self.lock)
                    futures += [future]
                else:
                    future = executor.submit(run_train, trainer, dyn_data, power_labels, pipeline_lock=self.lock)
                    futures += [future]
            self.print_log('Waiting for {} trainers to complete...'.format(len(futures)))
            wait(futures)
            self.print_log('{}/{} trainers are trained from {} to {}'.format(len(futures), len(self.trainers), feature_group, energy_source))

    def process(self, input_query_results, energy_components, energy_source, feature_group, aggr=True):
        print("========================================================", flush=True)
        self.print_log("{} start processing.".format(feature_group))
        query_results = input_query_results.copy()
        # 1. get abs_data
        extracted_data, power_labels = self.get_abs_data(query_results, energy_components, feature_group, energy_source, aggr=aggr)
        if extracted_data is None:
            self.print_log("cannot extract data")
            return False, None, None
        self.print_log("{} extraction done.".format(feature_group))
        abs_data = extracted_data.copy()
        # 2. get dyn_data
        isolated_data  = self.get_dyn_data(query_results, energy_components, feature_group, energy_source)
        if isolated_data is None:
            self.print_log("cannot isolate data")
            return False, None, None
        self.print_log("{} isolation done.".format(feature_group))
        dyn_data = isolated_data.copy()
        self._train(abs_data, dyn_data, power_labels, energy_source, feature_group)
        print("========================================================", flush=True)
        return True, abs_data, dyn_data
    
    def process_multiple_query(self, input_query_results_list, energy_components, energy_source, feature_group, aggr=True):
        # 1. get abs_data
        index = 0
        abs_data_list = []
        for input_query_results in input_query_results_list:
            query_results = input_query_results.copy()
            extracted_data, power_labels = self.get_abs_data(query_results, energy_components, feature_group, energy_source, aggr=aggr)
            if extracted_data is None:
                self.print_log("cannot extract data index={}".format(index))
                continue
            abs_data_list += [extracted_data]
            index += 1
        if len(abs_data_list) == 0:
            self.print_log("cannot get abs data")
            return False, None, None
        abs_data = pd.concat(abs_data_list)
        self.print_log("{} extraction done.".format(feature_group))
        # 2. get dyn_data
        index = 0
        dyn_data_list = []
        for input_query_results in input_query_results_list:
            query_results = input_query_results.copy()
            isolated_data  = self.get_dyn_data(query_results, energy_components, feature_group, energy_source)
            if isolated_data is None:
                self.print_log("cannot isolate data index={}".format(index))
                continue
            dyn_data_list += [isolated_data]
            index += 1
        if len(dyn_data_list) == 0:
            self.print_log("cannot get dyn data")
            return False, None, None
        dyn_data = pd.concat(dyn_data_list)
        self.print_log("{} isolation done.".format(feature_group))
        self._train(abs_data, dyn_data, power_labels, energy_source, feature_group)
        print("========================================================", flush=True)
        return True, abs_data, dyn_data

    def print_log(self, message):
        print("{} pipeline: {}".format(self.name, message), flush=True)

    def save_metadata(self):
        all_metadata = get_all_metadata(model_toppath, self.name)
        for energy_source, model_type_metadata in all_metadata.items():
            for model_type, metadata_df in model_type_metadata.items():
                metadata_df = metadata_df.sort_values(by=[ERROR_KEY])
                save_pipeline_metadata(self.path, energy_source, model_type, metadata_df)
            
def initial_trainers(trainer_names, node_level, pipeline_name, target_energy_sources, valid_feature_groups):
    trainers = []
    for energy_source in target_energy_sources:
        energy_components = PowerSourceMap[energy_source]
        for feature_group in valid_feature_groups:
            for trainer_name in trainer_names:
                trainer_class = load_class("trainer", trainer_name)
                trainer = trainer_class(energy_components, feature_group.name, energy_source, node_level, pipeline_name=pipeline_name)
                trainers += [trainer]
    return trainers

def NewPipeline(pipeline_name, abs_trainer_names, dyn_trainer_names, extractor=DefaultExtractor(), isolator=MinIdleIsolator(), target_energy_sources=PowerSourceMap.keys(), valid_feature_groups=FeatureGroups.keys()):
    abs_trainers = initial_trainers(abs_trainer_names, node_level=True, pipeline_name=pipeline_name, target_energy_sources=target_energy_sources, valid_feature_groups=valid_feature_groups)
    dyn_trainers = initial_trainers(dyn_trainer_names, node_level=False, pipeline_name=pipeline_name, target_energy_sources=target_energy_sources, valid_feature_groups=valid_feature_groups)
    trainers = abs_trainers + dyn_trainers
    return Pipeline(pipeline_name, trainers, extractor, isolator)
    
