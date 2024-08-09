# extractor_test.py
# - extractor.extract
#
# To use response:
# from extractor_test import get_extract_results
# extract_results = get_extract_results(extractor_name, feature_group, node_level)

# import external src
import os

from kepler_model.train.pipeline import load_class
from kepler_model.train import DefaultExtractor, SmoothExtractor
from kepler_model.util.extract_types import component_to_col
from kepler_model.util.prom_types import node_info_column
from kepler_model.util.train_types import all_feature_groups
from kepler_model.util import FeatureGroups, FeatureGroup, PowerSourceMap
from kepler_model.util import assure_path, get_valid_feature_group_from_queries
from kepler_model.util import save_csv, load_csv

from tests.prom_test import get_query_results


data_path = os.path.join(os.path.dirname(__file__), "data")
assure_path(data_path)
extractor_output_path = os.path.join(data_path, "extractor_output")
assure_path(extractor_output_path)

if not os.path.exists(extractor_output_path):
    os.mkdir(extractor_output_path)

test_extractors = [DefaultExtractor(), SmoothExtractor()]

test_energy_source = "rapl-sysfs"
test_energy_components = PowerSourceMap[test_energy_source]
test_num_of_unit = 2
test_customize_extractors = []


def get_filename(extractor_name, feature_group, node_level):
    return "{}_{}_{}".format(extractor_name, feature_group, node_level)


def get_extract_result(extractor_name, feature_group, node_level, save_path=extractor_output_path):
    filename = get_filename(extractor_name, feature_group, node_level)
    return load_csv(save_path, filename)


def get_extract_results(extractor_name, node_level, save_path=extractor_output_path):
    all_results = dict()
    for feature_group in all_feature_groups:
        result = get_extract_result(extractor_name, feature_group, node_level, save_path=save_path)
        if result is not None:
            all_results[feature_group] = result
    return all_results


def save_extract_results(instance, feature_group, extracted_data, node_level, save_path=extractor_output_path):
    extractor_name = instance.__class__.__name__
    filename = get_filename(extractor_name, feature_group, node_level)
    save_csv(save_path, filename, extracted_data)


def get_expected_power_columns(energy_components=test_energy_components, num_of_unit=test_num_of_unit):
    # TODO: if ratio applied,
    # return [component_to_col(component, "package", unit_val) for component in energy_components for unit_val in range(0,num_of_unit)]
    return [component_to_col(component) for component in energy_components]


def assert_extract(extracted_data, power_columns, energy_components, num_of_unit, feature_group):
    extracted_data_column_names = extracted_data.columns
    # basic assert
    assert extracted_data is not None, "extracted data is None"
    assert len(power_columns) > 0, "no power label column {}".format(extracted_data_column_names)
    assert node_info_column in extracted_data_column_names, "no {} in column {}".format(node_info_column, extracted_data_column_names)
    # TODO: if ratio applied, expected_power_column_length = len(energy_components) * num_of_unit
    expected_power_column_length = len(energy_components)
    # detail assert
    assert len(power_columns) == expected_power_column_length, "unexpected power label columns {}, expected {}".format(power_columns, expected_power_column_length)
    # TODO: if ratio applied, expected_col_size must + 1 for power_ratio
    expected_col_size = expected_power_column_length + len(FeatureGroups[FeatureGroup[feature_group]]) + num_of_unit  # power ratio
    assert len(extracted_data_column_names) == expected_col_size, "unexpected column length: expected {}, got {}({}) ".format(expected_col_size, extracted_data_column_names, len(extracted_data_column_names))


def process(query_results, feature_group, save_path=extractor_output_path, customize_extractors=test_customize_extractors, energy_source=test_energy_source, num_of_unit=2):
    energy_components = PowerSourceMap[energy_source]
    global test_extractors
    for extractor_name in customize_extractors:
        test_extractors += [load_class("extractor", extractor_name)]
    for test_instance in test_extractors:
        extracted_data, power_columns, corr, _ = test_instance.extract(query_results, energy_components, feature_group, energy_source, node_level=True)
        assert_extract(extracted_data, power_columns, energy_components, num_of_unit, feature_group)
        save_extract_results(test_instance, feature_group, extracted_data, True, save_path=save_path)
        extracted_data, power_columns, corr, _ = test_instance.extract(query_results, energy_components, feature_group, energy_source, node_level=False)
        assert_extract(extracted_data, power_columns, energy_components, num_of_unit, feature_group)
        save_extract_results(test_instance, feature_group, extracted_data, False, save_path=save_path)
        print("Correlations:\n")
        print(corr)


def test_extractor_process():
    query_results = get_query_results()
    assert len(query_results) > 0, "cannot read_sample_query_results"
    valid_feature_groups = get_valid_feature_group_from_queries(query_results.keys())
    for fg in valid_feature_groups:
        feature_group = fg.name
        process(query_results, feature_group)
