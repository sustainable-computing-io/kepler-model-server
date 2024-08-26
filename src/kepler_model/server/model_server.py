import codecs
import enum
import logging
import os
import shutil
import sys

import click
import requests
from flask import Flask, json, make_response, request, send_file

from kepler_model.train import NodeTypeIndexCollection, NodeTypeSpec
from kepler_model.util.config import (
    ERROR_KEY,
    MODEL_SERVER_MODEL_LIST_PATH,
    MODEL_SERVER_MODEL_REQ_PATH,
    download_path,
    getConfig,
    initial_pipeline_urls,
    model_toppath,
)
from kepler_model.util.loader import (
    CHECKPOINT_FOLDERNAME,
    METADATA_FILENAME,
    any_node_type,
    default_pipelines,
    get_archived_file,
    get_largest_candidates,
    get_model_group_path,
    get_node_type_from_name,
    get_pipeline_path,
    is_matched_type,
    is_valid_model,
    load_json,
    load_weight,
    parse_filters,
)
from kepler_model.util.saver import WEIGHT_FILENAME
from kepler_model.util.train_types import (
    FeatureGroup,
    FeatureGroups,
    ModelOutputType,
    PowerSourceMap,
    convert_enery_source,
    get_valid_feature_groups,
    weight_support_trainers,
)

logger = logging.getLogger(__name__)

###############################################
# model request #


class ModelRequest:
    def __init__(self, metrics, output_type, source="rapl-sysfs", node_type=-1, weight=False, trainer_name="", filter="", pipeline_name="", spec=None, loose_node_type=True):
        # target source of power metric to be predicted (e.g., rapl-sysfs, acpi)
        self.source = convert_enery_source(source)
        # type of node to select a model learned from similar nodes (default: -1, applied universal model learned by all node_type (TODO))
        self.node_type = int(node_type) if node_type or node_type == 0 else -1
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
        self.loose_node_type = loose_node_type

# ModelListParams defines parameters for /best-models API
class ModelListParam(enum.Enum):
    EnergySource = "energy-source"
    OutputType = "output-type"
    FeatureGroup = "feature-group"
    NodeType = "node-type"
    Filter = "filter"

###########################################
MODEL_SERVER_PORT = int(getConfig("MODEL_SERVER_PORT", "8100"))

# pipelineName and nodeCollection are global dict values set at initial state (load_init_pipeline)
## pipelineName: map of energy_source to target pipeline name
pipelineName = dict()
## nodeCollection: map of pipeline_name to its node_collection, used for determining covering node_type of requesting node spec
nodeCollection = dict()

"""
select_best_model:
1. list model_names from valid_group_path (determined by valid features)
2. filter weight-supported model if requesting for model weight
3. filter matched type by requesting node_type or node_collection over node spec
4. if no candidate left, list model with largest number of cores
5. if fail to list, use all models from step 2
7. for each model, check validity and load
8. return the best model (lowest error)
"""


def select_best_model(spec, valid_group_path: str, filters: dict, energy_source: str, pipeline_name: str="", trainer_name: str="", node_type: int=any_node_type, weight: bool=False, loose_node_type: bool=True):
    # Set default pipeline if not specified
    if pipeline_name == "" and energy_source in default_pipelines:
        pipeline_name = default_pipelines[energy_source]

    # Find initial model list filtered by trainer
    initial_model_names = [f for f in os.listdir(valid_group_path) if f != CHECKPOINT_FOLDERNAME and not os.path.isfile(os.path.join(valid_group_path, f)) and os.path.exists(os.path.join(valid_group_path, f, METADATA_FILENAME + ".json")) and (trainer_name == "" or trainer_name in f)]
    if node_type != any_node_type:
        model_names = [name for name in initial_model_names if f"_{node_type}" in name]
        if len(model_names) == 0:
            if not loose_node_type:
                return None, None
            logger.warning(f"{valid_group_path} has no matched model for node type={node_type}, try all available models")
            model_names = initial_model_names
    else:
        model_names = initial_model_names

    # Filter weight models
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
        logger.info(f"no matched models; selecting from large candidates: {candidates}")
        if candidates is None:
            logger.warning("no large candidates; selecting from all available")
            candidates = model_names
    for model_name in candidates:
        model_savepath = os.path.join(valid_group_path, model_name)
        metadata = load_json(model_savepath, METADATA_FILENAME)
        if metadata is None or not is_valid_model(metadata, filters) or ERROR_KEY not in metadata:
            # invalid metadata
            logger.warning(f"invalid metadata {is_valid_model(metadata, filters)} : {metadata}")
            continue
        if weight:
            response = load_weight(model_savepath)
            if response is None:
                # fail to get weight file
                logger.warning(f"weight failed: {model_savepath}")
                continue
        else:
            response = get_archived_file(valid_group_path, model_name)
            if not os.path.exists(response):
                # archived model file does not exists
                logger.warning(f"archive failed: {response}")
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
    logger.info(f"get request /model: {model_request}")
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
    best_uncertainty = None
    best_looseness = None
    # find best model comparing best candidate from each valid feature group complied with filtering conditions
    for fg in valid_fgs:
        pipeline_name = pipelineName[energy_source]
        valid_group_path = get_model_group_path(model_toppath, output_type, fg, energy_source, pipeline_name=pipelineName[energy_source])
        node_type = req.node_type
        if req.node_type == any_node_type and req.spec is not None and not req.spec.is_none() and pipeline_name in nodeCollection:
            node_type, uncertainty, looseness = nodeCollection[pipeline_name].get_node_type(req.spec, loose_search=True)
        else:
            uncertainty = 0
            looseness = 0
        if os.path.exists(valid_group_path):
            best_candidate, response = select_best_model(req.spec, valid_group_path, filters, energy_source, req.pipeline_name, req.trainer_name, node_type, req.weight, loose_node_type=req.loose_node_type)
            if best_candidate is None:
                continue
            if node_type != any_node_type and best_model is not None and get_node_type_from_name(best_model['model_name']) == node_type:
                if get_node_type_from_name(best_candidate['model_name']) != node_type:
                    continue
            if best_model is None or best_model[ERROR_KEY] > best_candidate[ERROR_KEY]:
                best_model = best_candidate
                best_response = response
                best_uncertainty = uncertainty
                best_looseness = looseness
    if best_model is None:
        return make_response(f"cannot find model for {model_request} at the moment", 400)
    logger.info(f"response: model {best_model['model_name']} by {best_model['features']} with {ERROR_KEY}={best_model[ERROR_KEY]} selected with uncertainty={best_uncertainty}, looseness={best_looseness}")
    if req.weight:
        try:
            response = app.response_class(response=json.dumps(best_response), status=200, mimetype="application/json")
            return response
        except ValueError as err:
            return make_response(f"get weight response error: {err}", 400)
    else:
        try:
            return send_file(best_response, as_attachment=True)
        except ValueError as err:
            return make_response(f"send archived model error: {err}", 400)


# get_available_models: return name list of best-candidate pipelines
@app.route(MODEL_SERVER_MODEL_LIST_PATH, methods=["GET"])
def get_available_models():
    fg = request.args.get(ModelListParam.FeatureGroup.value)
    ot = request.args.get(ModelListParam.OutputType.value)
    energy_source = request.args.get(ModelListParam.EnergySource.value)
    energy_source = convert_enery_source(energy_source)
    node_type = request.args.get(ModelListParam.NodeType.value)
    filter = request.args.get(ModelListParam.Filter.value)

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

        if node_type is None:
            node_type = any_node_type
        else:
            node_type = int(node_type)

        if filter is None:
            filters = dict()
        else:
            filters = parse_filters(filter)

        model_names = dict()
        for output_type in output_types:
            logger.debug(f"Searching output type {output_type}")
            model_names[output_type.name] = dict()
            for fg in valid_fgs:
                logger.debug(f"Searching feature group {fg}")
                valid_group_path = get_model_group_path(model_toppath, output_type, fg, energy_source, pipeline_name=pipelineName[energy_source])
                if os.path.exists(valid_group_path):
                    best_candidate, _ = select_best_model(None, valid_group_path, filters, energy_source, node_type=node_type, loose_node_type=False)
                    if best_candidate is None:
                        continue
                    model_names[output_type.name][fg.name] = best_candidate["model_name"]
        response = app.response_class(response=json.dumps(model_names), status=200, mimetype="application/json")
        return response
    except (ValueError, Exception) as err:
        return make_response(f"failed to get best model list: {err}", 400)


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
        logger.info(f"initial pipeline is loaded to {pipeline_path}")
        for energy_source in PowerSourceMap.keys():
            if os.path.exists(os.path.join(pipeline_path, energy_source)):
                pipelineName[energy_source] = pipeline_name
                logger.info(f"set pipeline {pipeline_name} for {energy_source}")


# load_init_pipeline: load pipeline from URLs and set pipeline variables
def load_init_pipeline():
    for initial_pipeline_url in initial_pipeline_urls:
        logger.info(f"downloading archived pipeline from URL: {initial_pipeline_url}")
        response = requests.get(initial_pipeline_url)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"response: {response}")

        if response.status_code != 200:
            logger.error(f"failed to download archived pipeline - status code: {response.status_code}, url: {initial_pipeline_url}")
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
            logger.error(f"failed to unpack downloaded pipeline: {e}")
            return
        # remove downloaded zip
        os.remove(tmp_filepath)
    set_pipelines()


@click.command()
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["debug", "info", "warn", "error"]),
    default="info",
    required=False,
)
def run(log_level: str):
    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)
    load_init_pipeline()
    app.run(host="0.0.0.0", port=MODEL_SERVER_PORT)


if __name__ == "__main__":
    sys.exit(run())
