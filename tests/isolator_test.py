import os

import numpy as np

from kepler_model.train import (
    DefaultProfiler,
    MinIdleIsolator,
    NoneIsolator,
    ProfileBackgroundIsolator,
    TrainIsolator,
    generate_profiles,
)
from kepler_model.train.extractor.preprocess import find_correlations
from kepler_model.util import FeatureGroup, FeatureGroups, assure_path, load_csv, save_csv
from kepler_model.util.extract_types import container_level_index, node_level_index
from kepler_model.util.prom_types import prom_responses_to_results
from kepler_model.util.train_types import all_feature_groups
from tests.extractor_test import (
    extractor_output_path,
    get_expected_power_columns,
    get_extract_results,
    test_energy_source,
    test_extractors,
)
from tests.prom_test import get_prom_response

isolator_output_path = os.path.join(os.path.dirname(__file__), "data", "isolator_output")
assure_path(isolator_output_path)

test_idle_response = get_prom_response(save_name="idle")
test_idle_data = prom_responses_to_results(test_idle_response)
profile_map = DefaultProfiler.process(test_idle_data)
test_profiles = generate_profiles(profile_map)
test_isolators = [MinIdleIsolator(), NoneIsolator()]


def get_filename(isolator_name, extractor_name, feature_group):
    return f"{isolator_name}_{extractor_name}_{feature_group}_{False}"


def get_isolate_result(isolator_name, extractor_name, feature_group, save_path=isolator_output_path):
    filename = get_filename(isolator_name, extractor_name, feature_group)
    return load_csv(save_path, filename)


def get_isolate_results(isolator_name, extractor_name, save_path=isolator_output_path):
    all_results = dict()
    for feature_group in all_feature_groups:
        result = get_isolate_result(isolator_name, extractor_name, feature_group, save_path=save_path)
        if result is not None:
            all_results[feature_group] = result
    return all_results


def save_results(instance, extractor_name, feature_group, isolated_data, save_path=isolator_output_path):
    filename = get_filename(instance.__class__.__name__, extractor_name, feature_group)
    save_csv(save_path, filename, isolated_data)


def assert_isolate(extractor_result, isolated_data):
    isolated_data_column_names = isolated_data.columns
    assert isolated_data is not None, "isolated data is None"
    value_df = isolated_data.reset_index().drop(columns=container_level_index)
    negative_df = value_df[(value_df < 0).all(1)]
    assert len(negative_df) == 0, f"all data must be non-negative \n {negative_df}"
    assert len(extractor_result.columns) == len(
        isolated_data_column_names
    ), f"unexpected column length: expected {len(extractor_result.columns)}, got {isolated_data_column_names}({len(isolated_data_column_names)}) "


def find_correlation_of_isolated_data(isolated_data, workload_features, energy_source=test_energy_source, power_columns=get_expected_power_columns()):
    feature_power_data = isolated_data.groupby(node_level_index).sum()
    corr = find_correlations(energy_source, feature_power_data, power_columns, workload_features)
    return corr


def get_max_corr(isolated_data, workload_features, energy_source=test_energy_source, power_columns=get_expected_power_columns()):
    corr_data = find_correlation_of_isolated_data(isolated_data, workload_features, energy_source=energy_source, power_columns=power_columns)
    try:
        corr = corr_data.values.max()
        if np.isnan(corr) or corr < 0:
            corr = 0
    except Exception as e:
        print(e)
    return corr


def process(test_isolators=test_isolators, customize_isolators=[], extract_path=extractor_output_path, save_path=isolator_output_path):
    for custom_isolator in customize_isolators:
        test_isolators += [custom_isolator]
    for test_instance in test_isolators:
        isolator_name = test_instance.__class__.__name__
        for extractor in test_extractors:
            extractor_name = extractor.__class__.__name__
            extractor_results = get_extract_results(extractor_name, node_level=False, save_path=extract_path)
            for feature_group, extract_result in extractor_results.items():
                print(f"{isolator_name} isolate {extractor_name}_{feature_group}")
                isolated_data = test_instance.isolate(extract_result, label_cols=get_expected_power_columns(), energy_source=test_energy_source)
                workload_features = FeatureGroups[FeatureGroup[feature_group]]
                corr = find_correlation_of_isolated_data(isolated_data, workload_features)
                print(corr)
                assert_isolate(extract_result, isolated_data)
                save_results(test_instance, extractor_name, feature_group, isolated_data, save_path=save_path)


def test_isolators_process():
    # Add customize isolator here
    customize_isolators = [ProfileBackgroundIsolator(test_profiles, test_idle_data)]
    customize_isolators += [TrainIsolator(idle_data=test_idle_data, profiler=DefaultProfiler)]
    customize_isolators += [TrainIsolator(target_hints=["stress"])]
    process(customize_isolators=customize_isolators)
