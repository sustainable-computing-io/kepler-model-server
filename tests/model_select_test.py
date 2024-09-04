## Test Server API (model selection)
# requires model-server
import codecs
import os
import shutil

import requests

from kepler_model.estimate.model_server_connector import list_all_models
from kepler_model.server.model_server import MODEL_SERVER_PORT
from kepler_model.train.profiler.node_type_index import NodeTypeIndexCollection, NodeTypeSpec
from kepler_model.util.config import default_pipelines, download_path
from kepler_model.util.loader import base_model_url, load_metadata, load_remote_json
from kepler_model.util.saver import NODE_TYPE_INDEX_FILENAME
from kepler_model.util.similarity import compute_jaccard_similarity
from kepler_model.util.train_types import FeatureGroup, FeatureGroups, ModelOutputType, NodeAttribute
from tests.model_server_test import get_model_request_json

TMP_FILE = "download.zip"
test_data_path = os.path.join(os.path.dirname(__file__), "data")

# set environment
os.environ["MODEL_SERVER_URL"] = "http://localhost:8100"
test_energy_sources = ["rapl-sysfs"]


def get_node_types(energy_source):
    pipeline_name = default_pipelines[energy_source]
    url_path = os.path.join(base_model_url, pipeline_name, NODE_TYPE_INDEX_FILENAME)
    return load_remote_json(url_path)


def make_request_with_spec(metrics, output_type, node_type=-1, trainer_name="", energy_source="rapl-sysfs", spec=None):
    weight = False
    model_request = get_model_request_json(metrics, output_type, node_type, weight, trainer_name, energy_source)
    model_request["machine_spec"] = spec
    response = requests.post(f"http://localhost:{MODEL_SERVER_PORT}/model", json=model_request)
    assert response.status_code == 200, response.text
    output_path = os.path.join(download_path, output_type.name)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    with codecs.open(TMP_FILE, "wb") as f:
        f.write(response.content)
    shutil.unpack_archive(TMP_FILE, output_path)
    metadata = load_metadata(output_path)
    os.remove(TMP_FILE)
    return metadata["model_name"], metadata["features"]


def check_select_model(model_name, features, best_model_map):
    assert model_name != "", "model name should not be empty."
    found = False
    for cmp_fg_name, expected_best_model_without_node_type in best_model_map.items():
        cmp_metrics = FeatureGroups[FeatureGroup[cmp_fg_name]]
        if cmp_metrics == features:
            found = True
            assert (
                model_name == expected_best_model_without_node_type
            ), f"should select best model {expected_best_model_without_node_type} (select {model_name})"
            break
    assert found, f"must found matched best model without node_type for {features}: {best_model_map}"


def process(node_type, info, output_type, energy_source, valid_fgs, best_model_by_source):
    expected_suffix = f"_{node_type}"
    for fg_name in valid_fgs.keys():
        metrics = FeatureGroups[FeatureGroup[fg_name]]
        model_name, features = make_request_with_spec(metrics, output_type, energy_source=energy_source)
        check_select_model(model_name, features, best_model_by_source)
        model_name, features = make_request_with_spec(metrics, output_type, node_type=node_type, energy_source=energy_source)
        assert expected_suffix in model_name, "model must be a matched type"
        check_select_model(model_name, features, valid_fgs)
        model_name, features = make_request_with_spec(metrics, output_type, spec=info["attrs"], energy_source=energy_source)
        assert expected_suffix in model_name, "model must be a matched type"
        check_select_model(model_name, features, valid_fgs)
        fixed_some_spec = {"processor": info["attrs"]["processor"], "memory": info["attrs"]["memory"]}
        model_name, features = make_request_with_spec(metrics, output_type, spec=fixed_some_spec, energy_source=energy_source)
        assert expected_suffix in model_name, "model must be a matched type"
        check_select_model(model_name, features, valid_fgs)
        uncovered_spec = info["attrs"].copy()
        uncovered_spec["processor"] = "_".join(uncovered_spec["processor"].split("_")[:-1])
        model_name, features = make_request_with_spec(metrics, output_type, spec=uncovered_spec, energy_source=energy_source)
        assert expected_suffix in model_name, "model must be a matched type"
        check_select_model(model_name, features, valid_fgs)


def test_process():
    # test getting model from server
    os.environ["MODEL_SERVER_ENABLE"] = "true"
    available_models = list_all_models()
    assert len(available_models) > 0, "must have more than one available models"
    print("Available Models:", available_models)
    for energy_source in test_energy_sources:
        node_types = get_node_types(energy_source)
        best_model_by_source_map = list_all_models(energy_source=energy_source)
        for node_type, info in node_types.items():
            available_models = list_all_models(node_type=node_type, energy_source=energy_source)
            if len(available_models) > 0:
                for output_type_name, valid_fgs in available_models.items():
                    output_type = ModelOutputType[output_type_name]
                    process(node_type, info, output_type, energy_source, valid_fgs, best_model_by_source=best_model_by_source_map[output_type_name])
            else:
                print(f"skip {energy_source}/{node_type} because on available models")


def test_get_node_type():
    node_collection = NodeTypeIndexCollection(test_data_path)
    for index, spec in node_collection.node_type_index.items():
        cmp_spec = spec.copy()
        cmp_index, uncertainty, looseness = node_collection.get_node_type(cmp_spec)
        assert cmp_index == index
        assert uncertainty == 0
        assert looseness == 0
    empty_spec = NodeTypeSpec()
    cmp_index, uncertainty, looseness = node_collection.get_node_type(empty_spec)
    assert cmp_index != -1
    assert looseness == 0
    empty_spec.attrs[NodeAttribute.PROCESSOR] = "unconvered"
    cmp_index, uncertainty, looseness = node_collection.get_node_type(empty_spec)
    assert cmp_index == -1
    cmp_index, uncertainty, looseness = node_collection.get_node_type(empty_spec, loose_search=True)
    assert cmp_index != -1
    assert looseness > 0


def test_similarity_computation():
    for energy_source in test_energy_sources:
        node_types = get_node_types(energy_source)
        for node_type, info in node_types.items():
            spec = NodeTypeSpec(**info["attrs"])
            for cmp_node_type, cmp_info in node_types.items():
                cmp_spec = NodeTypeSpec(**cmp_info["attrs"])
                similarity = spec.get_similarity(cmp_spec, debug=True)
                if node_type == cmp_node_type:
                    assert similarity == 1, "similarity must be one for the same type"
                else:
                    assert similarity >= 0, f"similarity must be >= 0, {node_type}-{cmp_node_type} ({similarity})"
                    assert similarity <= 1, f"similarity must be <= 1, {node_type}-{cmp_node_type} ({similarity})"


def test_compute_jaccard_similarity():
    testcases = {
        ("", ""): 1,
        ("", "some"): 0,
        ("intel_xeon_platinum_8259cl", "intel_xeon_platinum_8259cl"): 1,
        ("abcd", "abcdefgh"): 0.25,
    }
    for tc, value in testcases.items():
        assert compute_jaccard_similarity(tc[0], tc[1]) == value


if __name__ == "__main__":
    test_process()
    test_similarity_computation()
    test_compute_jaccard_similarity()
