import os
import sys
import threading

util_path = os.path.join(os.path.dirname(__file__), '..', 'util')
sys.path.append(util_path)

from extractor import DefaultExtractor
from isolator import MinIdleIsolator

from util import PowerSourceMap, FeatureGroups

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

    def process(self, input_query_results, energy_components, feature_group, energy_source, aggr=True):
        query_results = input_query_results.copy()
        extracted_data, power_labels, corr = self.extractor.extract(query_results, energy_components, feature_group, energy_source, node_level=True, aggr=aggr)
        if extracted_data is None:
            self.print_log("cannot extract data")
            return False, None, None
        abs_data = extracted_data.copy()
        extracted_data, power_labels, corr = self.extractor.extract(query_results, energy_components, feature_group, energy_source, node_level=False)
        isolated_data = self.isolator.isolate(extracted_data, label_cols=power_labels, energy_source=energy_source)
        if isolated_data is None:
            self.print_log("cannot isolate data")
            return False, None, None
        dyn_data = isolated_data.copy()

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
        return True, abs_data, dyn_data

    def print_log(self, message):
        print("{} pipeline: {}".format(self.name, message))

def initial_trainers(profiles, trainer_names, node_level, pipeline_name, target_energy_sources, valid_feature_groups):
    trainers = []
    for energy_source in target_energy_sources:
        energy_components = PowerSourceMap[energy_source]
        for feature_group in valid_feature_groups:
            for trainer_name in trainer_names:
                trainer_class = load_class("trainer", trainer_name)
                trainer = trainer_class(profiles, energy_components, feature_group.name, energy_source, node_level, pipeline_name=pipeline_name)
                trainers += [trainer]
    return trainers

def NewPipeline(pipeline_name, profiles, abs_trainer_names, dyn_trainer_names, extractor=DefaultExtractor(), isolator=MinIdleIsolator(), target_energy_sources=PowerSourceMap.keys(), valid_feature_groups=FeatureGroups.keys()):
    abs_trainers = initial_trainers(profiles, abs_trainer_names, node_level=True, pipeline_name=pipeline_name, target_energy_sources=target_energy_sources, valid_feature_groups=valid_feature_groups)
    dyn_trainers = initial_trainers(profiles, dyn_trainer_names, node_level=False, pipeline_name=pipeline_name, target_energy_sources=target_energy_sources, valid_feature_groups=valid_feature_groups)
    trainers = abs_trainers + dyn_trainers
    return Pipeline(pipeline_name, trainers, extractor, isolator)
    
