import os
import sys
import time

util_path = os.path.join(os.path.dirname(__file__), '..', 'util')
sys.path.append(util_path)

from prom.query import PrometheusClient, PROM_QUERY_INTERVAL
from util.config import getConfig

SAMPLING_INTERVAL = PROM_QUERY_INTERVAL
SAMPLING_INTERVAL = getConfig('SAMPLING_INTERVAL', SAMPLING_INTERVAL)
SAMPLING_INTERVAL = int(SAMPLING_INTERVAL)

from train import NewPipeline, PowerSourceMap, FeatureGroups, load_class, load_all_profiles, DefaultExtractor, MinIdleIsolator, ProfileIsolator

default_trainers = ['GradientBoostingRegressorTrainer']
abs_trainer_names = default_trainers + []
dyn_trainer_names = default_trainers + []

def initial_trainers(profiles, trainer_names, node_level):
    trainers = []
    for energy_source, energy_components in PowerSourceMap.items():
        for feature_group in FeatureGroups.Keys():
            for trainer_name in trainer_names:
                trainer_class = load_class("trainer", trainer_name)
                trainer = trainer_class(profiles, energy_components, feature_group.name, energy_source, node_level)
                trainers += [trainer]
    return trainers

def initial_pipelines():
    profiles = load_all_profiles()
    pipeline_name = "DefaultPipeline"
    abs_trainers = initial_trainers(profiles, abs_trainer_names, node_level=True)
    dyn_trainers = initial_trainers(profiles, dyn_trainer_names, node_level=False)
    trainers = abs_trainers + dyn_trainers
    profile_pipeline = NewPipeline(pipeline_name, trainers, extractor=DefaultExtractor(), isolator=ProfileIsolator(profiles))
    non_profile_pipeline = NewPipeline(pipeline_name, trainers, extractor=DefaultExtractor(), isolator=MinIdleIsolator())
    return profile_pipeline, non_profile_pipeline

if __name__ == '__main__':
    profile_pipeline, non_profile_pipeline = initial_pipelines()
    prom_client = PrometheusClient()
    while True:
        prom_client.query()
        query_results = prom_client.snapshot_query_result()

        for energy_source, energy_components in PowerSourceMap.items():
            for feature_group in FeatureGroups.Keys():
                success, _, _ = profile_pipeline.process(query_results, energy_components, feature_group, energy_source)
                if not success:
                    # failed to process with profile, try non_profile pipeline
                    non_profile_pipeline.process(query_results, energy_components, feature_group, energy_source)

        time.sleep(SAMPLING_INTERVAL)