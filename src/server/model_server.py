from flask import Flask, request, json, make_response, send_file

import os
import sys
import codecs
import shutil
import requests

src_path = os.path.join(os.path.dirname(__file__), "..")
util_path = os.path.join(os.path.dirname(__file__), "util")

sys.path.append(src_path)
sys.path.append(util_path)

from util.train_types import get_valid_feature_groups, ModelOutputType, FeatureGroups, FeatureGroup, PowerSourceMap, weight_support_trainers
from util.config import getConfig, model_toppath, ERROR_KEY, MODEL_SERVER_MODEL_REQ_PATH, MODEL_SERVER_MODEL_LIST_PATH, initial_pipeline_urls, download_path
from util.loader import parse_filters, is_valid_model, load_json, load_weight, get_model_group_path, get_archived_file, METADATA_FILENAME, CHECKPOINT_FOLDERNAME, get_pipeline_path, any_node_type, is_matched_type, get_largest_candidates
from util.saver import WEIGHT_FILENAME
from train import NodeTypeSpec, NodeTypeIndexCollection

###############################################
# model request #


class ModelRequest:
    def __init__(self, metrics, output_type, source="rapl-sysfs", node_type=-1, weight=False, trainer_name="", filter="", pipeline_name="", spec=None):
        # target source of power metric to be predicted (e.g., rapl-sysfs, acpi)
        self.source = source
        # type of node to select a model learned from similar nodes (default: -1, applied universal model learned by all node_type (TODO))
        self.node_type = node_type
        # list of available resource usage metrics to find applicable models (using a valid feature group that can be obtained from the list)
        self.metrics = metrics
        # specific trainer name (default: empty, selecting any of the best trainer)
        self.trainer_name = trainer_name
        # filtering conditions from metadata attribute such as mae (default: empty, no filtering condition)
        self.filter = filter
        # Dyn (not include power at idling state, for container/process level) or Abs (includ power at idling state for node level)
        self.output_type = output_type
        # whether requesting just a linear regression weight or any model archive file (default: False, an archive file of any model)
        self.weight = weight
        # specific pipeline (default: empty, selecting default pipeline)
        self.pipeline_name = pipeline_name
        # spec of requesting node to determine node_type
        self.spec = NodeTypeSpec()
        if spec is not None:
            self.spec = NodeTypeSpec(**spec)


###########################################

MODEL_SERVER_PORT = 8100
MODEL_SERVER_PORT = getConfig("MODEL_SERVER_PORT", MODEL_SERVER_PORT)
MODEL_SERVER_PORT = int(MODEL_SERVER_PORT)

# pipelineName and nodeCollection are global dict values set at initial state (load_init_pipeline)
## pipelineName: map of energy_source to target pipeline name
pipelineName = dict()
## nodeCollection: map of pipeline_name to its node_collection, used for determining covering node_type of requesting node spec
nodeCollection = dict()

"""
select_best_model:
1. list model_names from valid_grouppath (determined by valid features)
2. filter weight-supported model if requesting for model weight
3. filter matched type by requesting node_type or node_collection over node spec
4. if no candidate left, list model with largest number of cores
5. if fail to list, use all models from step 2
7. for each model, check validity and load
8. return the best model (lowest error)
"""


def select_best_model(spec, valid_groupath, filters, energy_source, pipeline_name="", trainer_name="", node_type=any_node_type, weight=False):
    model_names = [f for f in os.listdir(valid_groupath) if f != CHECKPOINT_FOLDERNAME and not os.path.isfile(os.path.join(valid_groupath, f)) and (trainer_name == "" or trainer_name in f)]
    if weight:
        model_names = [name for name in model_names if name.split("_")[0] in weight_support_trainers]
    # Load metadata of trainers
    best_cadidate = None
    best_response = None
    candidates = []
    for model_name in model_names:
        if not is_matched_type(nodeCollection, spec, pipeline_name, model_name, node_type, energy_source):
            continue
        candidates += [model_name]
    if len(model_names) > 0 and len(candidates) == 0:
        # loosen all spec
        candidates = get_largest_candidates(model_names, pipeline_name, nodeCollection, energy_source)
        print("no matched models, select from large candidates: ", candidates)
        if candidates is None:
            print("no large candidates, select from all availables")
            candidates = model_names
    for model_name in candidates:
        model_savepath = os.path.join(valid_groupath, model_name)
        metadata = load_json(model_savepath, METADATA_FILENAME)
        if metadata is None or not is_valid_model(metadata, filters) or ERROR_KEY not in metadata:
            # invalid metadata
            print("invalid", is_valid_model(metadata, filters), metadata)
            continue
        if weight:
            response = load_weight(model_savepath)
            if response is None:
                # fail to get weight file
                print("weight failed", model_savepath)
                continue
        else:
            response = get_archived_file(valid_groupath, model_name)
            if not os.path.exists(response):
                # archived model file does not exists
                print("archived failed", response)
                continue
        if best_cadidate is None or best_cadidate[ERROR_KEY] > metadata[ERROR_KEY]:
            best_cadidate = metadata
            best_response = response
    return best_cadidate, best_response


app = Flask(__name__)


# get_model: return archive file or LR weight based on request (req)
@app.route(MODEL_SERVER_MODEL_REQ_PATH, methods=["POST"])
def get_model():
    model_request = request.get_json()
    print("get request /model: {}".format(model_request))
    req = ModelRequest(**model_request)
    energy_source = req.source
    # TODO: need revisit if get more than one rapl energy source
    if energy_source is None or "rapl" in energy_source:
        energy_source = "rapl-sysfs"

    # find valid feature groups from available metrics in request
    valid_fgs = get_valid_feature_groups(req.metrics)
    # parse filtering conditions of metadata attribute from request if exists  (e.g., minimum mae)
    filters = parse_filters(req.filter)
    output_type = ModelOutputType[req.output_type]
    best_model = None
    best_response = None
    # find best model comparing best candidate from each valid feature group complied with filtering conditions
    for fg in valid_fgs:
        valid_groupath = get_model_group_path(model_toppath, output_type, fg, energy_source, pipeline_name=pipelineName[energy_source])
        if os.path.exists(valid_groupath):
            best_candidate, response = select_best_model(req.spec, valid_groupath, filters, energy_source, req.pipeline_name, req.trainer_name, req.node_type, req.weight)
            if best_candidate is None:
                continue
            if best_model is None or best_model[ERROR_KEY] > best_candidate[ERROR_KEY]:
                best_model = best_candidate
                best_response = response
    if best_model is None:
        return make_response("cannot find model for {} at the moment".format(model_request), 400)
    if req.weight:
        try:
            response = app.response_class(response=json.dumps(best_response), status=200, mimetype="application/json")
            return response
        except ValueError as err:
            return make_response("get weight response error: {}".format(err), 400)
    else:
        try:
            return send_file(best_response, as_attachment=True)
        except ValueError as err:
            return make_response("send archived model error: {}".format(err), 400)


# get_available_models: return name list of best-candidate pipelines
@app.route(MODEL_SERVER_MODEL_LIST_PATH, methods=["GET"])
def get_available_models():
    fg = request.args.get("fg")
    ot = request.args.get("ot")
    energy_source = request.args.get("source")
    filter = request.args.get("filter")

    try:
        if fg is None:
            valid_fgs = [fg_key for fg_key in FeatureGroups.keys()]
        else:
            valid_fgs = [FeatureGroup[fg]]

        if ot is None:
            output_types = [ot for ot in ModelOutputType]
        else:
            output_types = [ModelOutputType[ot]]
        # TODO: need revisit if get more than one rapl energy source
        if energy_source is None or "rapl" in energy_source:
            energy_source = "rapl-sysfs"

        if filter is None:
            filters = dict()
        else:
            filters = parse_filters(filter)

        model_names = dict()
        for output_type in output_types:
            model_names[output_type.name] = dict()
            for fg in valid_fgs:
                valid_groupath = get_model_group_path(model_toppath, output_type, fg, energy_source, pipeline_name=pipelineName[energy_source])
                if os.path.exists(valid_groupath):
                    best_candidate, _ = select_best_model(None, valid_groupath, filters, energy_source)
                    if best_candidate is None:
                        continue
                    model_names[output_type.name][fg.name] = best_candidate["model_name"]
        response = app.response_class(response=json.dumps(model_names), status=200, mimetype="application/json")
        return response
    except (ValueError, Exception) as err:
        return make_response("failed to get best model list: {}".format(err), 400)


# upack_zip_files: unpack all model.zip files to model folder and copy model.json to model/weight.zip
def unpack_zip_files(root_folder):
    # Walk through all folders and subfolders
    for folder, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".zip"):
                zip_file_path = os.path.join(folder, file)
                extract_to = os.path.splitext(zip_file_path)[0]  # Extract to same location as the ZIP file
                # Make sure the destination folder exists, if not, create it
                if not os.path.exists(extract_to):
                    os.makedirs(extract_to)
                shutil.unpack_archive(zip_file_path, extract_to)
                weight_file = os.path.join(folder, file.replace(".zip", ".json"))
                if os.path.exists(weight_file):
                    shutil.copy(weight_file, os.path.join(extract_to, WEIGHT_FILENAME + ".json"))


# set_pipelines: set global pipeline variables, nodeCollection and pipelineName
def set_pipelines():
    pipeline_names = [f for f in os.listdir(model_toppath) if os.path.exists(os.path.join(model_toppath, f, METADATA_FILENAME + ".json"))]
    for pipeline_name in pipeline_names:
        pipeline_path = get_pipeline_path(model_toppath, pipeline_name=pipeline_name)
        global nodeCollection
        nodeCollection[pipeline_name] = NodeTypeIndexCollection(pipeline_path)
        print("initial pipeline is loaded to {}".format(pipeline_path))
        for energy_source in PowerSourceMap.keys():
            if os.path.exists(os.path.join(pipeline_path, energy_source)):
                pipelineName[energy_source] = pipeline_name
                print("set pipeline {} for {}".format(pipeline_name, energy_source))


# load_init_pipeline: load pipeline from URLs and set pipeline variables
def load_init_pipeline():
    for initial_pipeline_url in initial_pipeline_urls:
        print("try downloading archieved pipeline from URL: {}".format(initial_pipeline_url))
        response = requests.get(initial_pipeline_url)
        print(response)
        if response.status_code != 200:
            print("failed to download archieved pipeline.")
            return
        # delete existing default pipeline
        basename = os.path.basename(initial_pipeline_url)
        pipeline_name = basename.split(".zip")[0]
        pipeline_path = get_pipeline_path(model_toppath, pipeline_name=pipeline_name)
        if os.path.exists(pipeline_path):
            shutil.rmtree(pipeline_path)
        os.mkdir(pipeline_path)
        # unpack pipeline
        try:
            filename = basename
            tmp_filepath = os.path.join(download_path, filename)
            with codecs.open(tmp_filepath, "wb") as f:
                f.write(response.content)
            shutil.unpack_archive(tmp_filepath, pipeline_path)
            unpack_zip_files(pipeline_path)
        except Exception as e:
            print("failed to unpack downloaded pipeline: ", e)
            return
        # remove downloaded zip
        os.remove(tmp_filepath)
    set_pipelines()


def run():
    load_init_pipeline()
    app.run(host="0.0.0.0", port=MODEL_SERVER_PORT)


if __name__ == "__main__":
    run()
