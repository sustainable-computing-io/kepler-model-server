import os
import sys

src_path = os.path.join(os.path.dirname(__file__), '../src')
train_path = os.path.join(os.path.dirname(__file__), '../src/train')

sys.path.append(src_path)
sys.path.append(train_path)

from train import NewPipeline, load_class, load_all_profiles, MinIdleIsolator, PowerSourceMap
from extractor_test import read_sample_query_results, test_extractors, feature_groups
from isolator_test import test_isolators
from trainer_test import trainer_names, assert_train

def new_pipeline(profiles, feature_group, energy_source, energy_components, extractor, trainer_names, isolator=MinIdleIsolator()):
    trainers = []
    pipeline_name = "{}_{}".format(isolator.__class__.__name__, extractor.__class__.__name__)
    for trainer_name in trainer_names:
        trainer_class = load_class("trainer", trainer_name)
        abs_trainer = trainer_class(profiles, energy_components, feature_group, energy_source, node_level=True)
        dyn_trainer = trainer_class(profiles, energy_components, feature_group, energy_source, node_level=False)
        trainers += [abs_trainer, dyn_trainer]
    return NewPipeline(pipeline_name, trainers, extractor, isolator)

def assert_pipeline(pipeline, query_results, feature_group, energy_source, energy_components):
    success, abs_data, dyn_data = pipeline.process(query_results, energy_components, feature_group, energy_source)
    assert success, "failed to process pipeline {}".format(pipeline.name) 
    for trainer in pipeline.trainers:
        if trainer.node_level:
            assert_train(trainer, abs_data, energy_components)
        else:
            assert_train(trainer, dyn_data, energy_components)

if __name__ == '__main__':
    query_results = read_sample_query_results()
    for extractor in test_extractors:
        profiles = load_all_profiles()
        for isolator in test_isolators:
            for energy_source, energy_components in PowerSourceMap.items():
                for feature_group in feature_groups:
                    pipeline = new_pipeline(profiles, feature_group, energy_source, energy_components, extractor, trainer_names, isolator=isolator)
                    print("Test pipeline ", pipeline.name)
                    assert_pipeline(pipeline, query_results, feature_group, energy_source, energy_components)