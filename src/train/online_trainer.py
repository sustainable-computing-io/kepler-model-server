#TODO: test
import  os
import  sys
import  time

util_path = os.path.join(os.path.dirname(__file__), '..', 'util')
sys.path.append(util_path)
prom_path = os.path.join(os.path.dirname(__file__), 'prom')
sys.path.append(prom_path)
extractor_path = os.path.join(os.path.dirname(__file__), 'extractor')
sys.path.append(extractor_path)
isolator_path = os.path.join(os.path.dirname(__file__), 'isolator')
sys.path.append(isolator_path)

from prom_query import  PrometheusClient
from prom_types import  get_valid_feature_group_from_queries, PROM_QUERY_INTERVAL
from config import  getConfig
from loader import  default_train_output_pipeline

SAMPLING_INTERVAL = PROM_QUERY_INTERVAL
SAMPLING_INTERVAL = getConfig('SAMPLING_INTERVAL', SAMPLING_INTERVAL)
SAMPLING_INTERVAL = int(SAMPLING_INTERVAL)

from train_types import  PowerSourceMap, FeatureGroups
from pipeline import  NewPipeline
from extractor import  DefaultExtractor
from isolator import  MinIdleIsolator, ProfileBackgroundIsolator
from profiler.profiler import  load_all_profiles


default_trainers = ['GradientBoostingRegressorTrainer']
abs_trainer_names = default_trainers + []
dyn_trainer_names = default_trainers + []

def initial_pipelines():
    target_energy_sources = PowerSourceMap.keys()
    valid_feature_groups = FeatureGroups.keys()
    profiles = load_all_profiles()
    profile_pipeline = NewPipeline(default_train_output_pipeline, abs_trainer_names, dyn_trainer_names, extractor=DefaultExtractor(), isolator=ProfileBackgroundIsolator(profiles), target_energy_sources=target_energy_sources, valid_feature_groups=valid_feature_groups)
    non_profile_pipeline = NewPipeline(default_train_output_pipeline, abs_trainer_names, dyn_trainer_names, extractor=DefaultExtractor(), isolator=MinIdleIsolator(), target_energy_sources=target_energy_sources, valid_feature_groups=valid_feature_groups)
    return profile_pipeline, non_profile_pipeline

if __name__ == '__main__':
    profile_pipeline, non_profile_pipeline = initial_pipelines()
    prom_client = PrometheusClient()
    while True:
        prom_client.query()
        query_results = prom_client.snapshot_query_result()
        valid_feature_groups = get_valid_feature_group_from_queries(query_results.keys())
        for energy_source, energy_components in PowerSourceMap.items():
            for feature_group in valid_feature_groups:
                success, _, _ = profile_pipeline.process(query_results, energy_components, energy_source, feature_group=feature_group)
                if not success:
                    # failed to process with profile, try non_profile pipeline
                    success, _, _ = non_profile_pipeline.process(query_results, energy_components, energy_source, feature_group=feature_group)
                    if success:
                        non_profile_pipeline.save_metadata()
                else:
                    profile_pipeline.save_metadata()
        time.sleep(SAMPLING_INTERVAL)