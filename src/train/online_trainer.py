import os
import sys
import time

util_path = os.path.join(os.path.dirname(__file__), '..', 'util')
sys.path.append(util_path)

from prom.prom_query import PrometheusClient, PROM_QUERY_INTERVAL
from util.config import getConfig

SAMPLING_INTERVAL = PROM_QUERY_INTERVAL
SAMPLING_INTERVAL = getConfig('SAMPLING_INTERVAL', SAMPLING_INTERVAL)
SAMPLING_INTERVAL = int(SAMPLING_INTERVAL)

from train import NewPipeline, PowerSourceMap, FeatureGroups, load_class, load_all_profiles, DefaultExtractor, MinIdleIsolator, ProfileIsolator

default_trainers = ['GradientBoostingRegressorTrainer']
abs_trainer_names = default_trainers + []
dyn_trainer_names = default_trainers + []


def initial_pipelines():
    profiles = load_all_profiles()
    pipeline_name = "DefaultPipeline"
    profile_pipeline = NewPipeline(pipeline_name, abs_trainer_names, dyn_trainer_names, extractor=DefaultExtractor(), isolator=ProfileIsolator(profiles))
    non_profile_pipeline = NewPipeline(pipeline_name, abs_trainer_names, dyn_trainer_names, extractor=DefaultExtractor(), isolator=MinIdleIsolator())
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