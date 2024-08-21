from kepler_model.train import NewPipeline, NodeTypeSpec
from kepler_model.util import PowerSourceMap, get_valid_feature_group_from_queries
from kepler_model.util.loader import default_node_type, default_train_output_pipeline
from tests.extractor_test import test_energy_source, test_extractors
from tests.isolator_test import test_isolators
from tests.prom_test import get_query_results, prom_output_filename, prom_output_path
from tests.trainer_test import assert_train, test_trainer_names

# fake spec value
spec_values = {"processor": "test", "cores": 1, "chips": 1, "memory": -1, "frequency": -1}
spec = NodeTypeSpec(**spec_values)

test_energy_sources = ["acpi", "rapl-sysfs"]


def assert_pipeline(pipeline, query_results, feature_group, energy_source, energy_components):
    success, abs_data, dyn_data = pipeline.process(query_results, energy_components, energy_source, feature_group=feature_group.name, replace_node_type=default_node_type)
    assert success, f"failed to process pipeline {pipeline.name}"
    for trainer in pipeline.trainers:
        if trainer.feature_group == feature_group and trainer.energy_source == energy_source:
            if trainer.node_level:
                assert_train(trainer, abs_data, energy_components)
            else:
                assert_train(trainer, dyn_data, energy_components)


def process(
    save_pipeline_name=default_train_output_pipeline,
    prom_save_path=prom_output_path,
    prom_save_name=prom_output_filename,
    abs_trainer_names=test_trainer_names,
    dyn_trainer_names=test_trainer_names,
    extractors=test_extractors,
    isolators=test_isolators,
    target_energy_sources=[test_energy_source],
    valid_feature_groups=None,
):
    query_results = get_query_results(save_path=prom_save_path, save_name=prom_save_name)
    if valid_feature_groups is None:
        valid_feature_groups = get_valid_feature_group_from_queries(query_results.keys())
    for extractor in extractors:
        for isolator in isolators:
            pipeline = NewPipeline(save_pipeline_name, abs_trainer_names, dyn_trainer_names, extractor=extractor, isolator=isolator, target_energy_sources=target_energy_sources, valid_feature_groups=valid_feature_groups)
            global spec
            pipeline.node_collection.index_train_machine("test", spec)
            for energy_source in target_energy_sources:
                energy_components = PowerSourceMap[energy_source]
                for feature_group in valid_feature_groups:
                    assert_pipeline(pipeline, query_results, feature_group, energy_source, energy_components)
            # save metadata
            pipeline.save_metadata()
            # save node collection
            pipeline.node_collection.save()
            # save pipeline
            pipeline.archive_pipeline()


def test_process():
    process(target_energy_sources=test_energy_sources)
