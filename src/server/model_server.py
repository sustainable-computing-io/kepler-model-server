from flask import Flask, request, json, make_response, send_file

import os
import sys
import logging
import codecs
import shutil
import requests
src_path = os.path.join(os.path.dirname(__file__), '..')
util_path = os.path.join(os.path.dirname(__file__), 'util')

sys.path.append(src_path)
sys.path.append(util_path)

from util.train_types import get_valid_feature_groups, ModelOutputType, FeatureGroups, FeatureGroup
from util.config import getConfig, model_toppath, ERROR_KEY, MODEL_SERVER_MODEL_REQ_PATH, MODEL_SERVER_MODEL_LIST_PATH, initial_pipeline_url, download_path
from util.loader import parse_filters, is_valid_model, load_json, load_weight, get_model_group_path, get_archived_file, METADATA_FILENAME, CHECKPOINT_FOLDERNAME, get_pipeline_path, any_node_type, is_matched_type, lr_trainers

###############################################
# model request 

class ModelRequest():
    def __init__(self, metrics, output_type, source='intel_rapl', node_type=-1, weight=False, trainer_name="", filter=""):
        # target source of power metric to be predicted (e.g., intel_rapl, acpi)
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

###########################################

MODEL_SERVER_PORT = 8100
MODEL_SERVER_PORT = getConfig('MODEL_SERVER_PORT', MODEL_SERVER_PORT)
MODEL_SERVER_PORT = int(MODEL_SERVER_PORT)

def select_best_model(valid_groupath, filters, trainer_name="", node_type=any_node_type, weight=False):
    model_names = [f for f in os.listdir(valid_groupath) if \
                    f != CHECKPOINT_FOLDERNAME \
                    and not os.path.isfile(os.path.join(valid_groupath,f)) \
                    and (trainer_name == "" or trainer_name in f)]
    if weight:
        model_names = [name for name in model_names if name.split("_")[0] in lr_trainers]
    # Load metadata of trainers
    best_cadidate = None
    best_response = None
    for model_name in model_names:
        if not is_matched_type(model_name, node_type):
            continue
        model_savepath = os.path.join(valid_groupath, model_name)
        metadata = load_json(model_savepath, METADATA_FILENAME)
        if metadata is None or not is_valid_model(metadata, filters) or ERROR_KEY not in metadata:
            # invalid metadata
            continue
        if weight:
            response = load_weight(model_savepath)
            if response is None:
                # fail to get weight file
                continue
        else:
            response = get_archived_file(valid_groupath, model_name)
            if not os.path.exists(response):
                # archived model file does not exists
                continue
        if best_cadidate is None or best_cadidate[ERROR_KEY] > metadata[ERROR_KEY]:
            best_cadidate = metadata
            best_response = response
    return best_cadidate, best_response

app = Flask(__name__)

# return archive file or LR weight based on request (req)
@app.route(MODEL_SERVER_MODEL_REQ_PATH, methods=['POST'])
def get_model():
    model_request = request.get_json()
    print("get request /model: {}".format(model_request))
    req = ModelRequest(**model_request)
    energy_source = req.source
    # TODO: need revisit if get more than one rapl energy source
    if energy_source is None or 'rapl' in energy_source:
        energy_source = 'intel_rapl'

    # find valid feature groups from available metrics in request
    valid_fgs = get_valid_feature_groups(req.metrics)
    # parse filtering conditions of metadata attribute from request if exists  (e.g., minimum mae)
    filters = parse_filters(req.filter)
    output_type = ModelOutputType[req.output_type]
    best_model = None
    best_response = None
    # find best model comparing best candidate from each valid feature group complied with filtering conditions
    for fg in valid_fgs:
        valid_groupath = get_model_group_path(model_toppath, output_type, fg, energy_source)
        if os.path.exists(valid_groupath):
            best_candidate, response = select_best_model(valid_groupath, filters, req.trainer_name, req.node_type, req.weight)
            if best_candidate is None:
                continue
            if best_model is None or best_model[ERROR_KEY] > best_candidate[ERROR_KEY]:
                best_model = best_candidate
                best_response = response
    if best_model is None:
        return make_response("cannot find model for {} at the moment".format(model_request), 400)
    if req.weight:
        try:
            response = app.response_class(
            response=json.dumps(best_response),
            status=200,
            mimetype='application/json'
            )
            return response
        except ValueError as err:
            return make_response("get weight response error: {}".format(err), 400)
    else:
        try:
            return send_file(best_response, as_attachment=True)
        except ValueError as err:
            return make_response("send archived model error: {}".format(err), 400)

# return name list of best-candidate pipelines
@app.route(MODEL_SERVER_MODEL_LIST_PATH, methods=['GET'])
def get_available_models():
    fg = request.args.get('fg')
    ot = request.args.get('ot')
    energy_source = request.args.get('source')
    filter = request.args.get('filter')

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
        if energy_source is None or 'rapl' in energy_source:
            energy_source = 'intel_rapl'

        if filter is None:
            filters = dict()
        else: 
            filters = parse_filters(filter)

        model_names = dict()
        for output_type in output_types:
            model_names[output_type.name] = dict()
            for fg in valid_fgs:
                valid_groupath = get_model_group_path(model_toppath, output_type, fg, energy_source)
                if os.path.exists(valid_groupath):
                    best_candidate, _ = select_best_model(valid_groupath, filters)
                    if best_candidate is None:
                        continue
                    model_names[output_type.name][fg.name] = best_candidate['model_name']
        response = app.response_class(
            response=json.dumps(model_names),
            status=200,
            mimetype='application/json'
            )
        return response
    except (ValueError, Exception) as err:
        return make_response("failed to get best model list: {}".format(err), 400)

def load_init_pipeline():
    print("try downloading archieved pipeline from URL: {}".format(initial_pipeline_url))
    response = requests.get(initial_pipeline_url)
    print(response)
    if response.status_code != 200:
        print("failed to download archieved pipeline.")
        return
    
    # delete existing default pipeline
    default_pipeline = get_pipeline_path(model_toppath)
    if os.path.exists(default_pipeline):
        shutil.rmtree(default_pipeline)
    os.mkdir(default_pipeline)

    # unpack pipeline
    try:
        TMP_FILE = 'tmp.zip'
        tmp_filepath = os.path.join(download_path, TMP_FILE)
        with codecs.open(tmp_filepath, 'wb') as f:
            f.write(response.content)
        shutil.unpack_archive(tmp_filepath, default_pipeline)
    except Exception as e:
        print("failed to unpack downloaded pipeline: ", e)
        return

    # remove downloaded zip
    os.remove(tmp_filepath)
    print("initial pipeline is loaded to {}".format(default_pipeline))

if __name__ == '__main__':
   load_init_pipeline()
   app.run(host="0.0.0.0", port=MODEL_SERVER_PORT)
