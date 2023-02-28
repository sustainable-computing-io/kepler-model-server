import os
import sys

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

def run_train(trainer, data, power_labels):
    trainer.process(data, power_labels)

class Pipeline():
    def __init__(self, name, trainers, extractor, isolator):
        self.extractor = extractor
        self.isolator = isolator
        self.trainers = trainers
        self.name = name

    def process(self, input_query_results, energy_components, feature_group, energy_source, aggr=True):
        query_results = input_query_results.copy()
        extracted_data, power_labels = self.extractor.extract(query_results, energy_components, feature_group, energy_source, node_level=True, aggr=aggr)
        if extracted_data is None:
            self.print_log("cannot extract data")
            return False, None, None
        abs_data = extracted_data.copy()
        extracted_data, power_labels = self.extractor.extract(query_results, energy_components, feature_group, energy_source, node_level=False)
        isolated_data = self.isolator.isolate(extracted_data, energy_source, power_labels)
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
                    future = executor.submit(run_train, trainer, abs_data, power_labels)
                    futures += [future]
                else:
                    future = executor.submit(run_train, trainer, dyn_data, power_labels)
                    futures += [future]
            self.print_log('Waiting for {} trainers to complete...'.format(len(futures)))
            wait(futures)
            self.print_log('{}/{} trainers are trained!'.format(len(futures), len(self.trainers)))
        return True, abs_data, dyn_data

    def print_log(self, message):
        print("{} pipeline: {}".format(self.name, message))