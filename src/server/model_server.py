from flask import Flask, request, json, make_response, send_file

import os
import sys
src_path = os.path.join(os.path.dirname(__file__), '..')
util_path = os.path.join(os.path.dirname(__file__), 'util')
train_path = os.path.join(os.path.dirname(__file__), '..', 'train')

sys.path.append(src_path)
sys.path.append(util_path)
sys.path.append(train_path)

from train import get_valid_feature_groups, is_weight_output, ModelOutputType, FeatureGroups, FeatureGroup
from util.config import getConfig
from util.loader import parse_filters, is_valid_model, load_json, get_model_weight, get_model_group_path, get_archived_file, METADATA_FILENAME, CHECKPOINT_FOLDERNAME

###############################################
# model request 

class ModelRequest():
    def __init__(self, metrics, output_type, source='rapl', node_type=-1, weight=False, trainer_name="", filter=""):
        # target source of power metric to be predicted (e.g., rapl, acpi)
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

ERROR_KEY = 'mae'
ERROR_KEY = getConfig('ERROR_KEY', ERROR_KEY)
MODEL_SERVER_PORT = 8100
MODEL_SERVER_PORT = getConfig('MODEL_SERVER_PORT', MODEL_SERVER_PORT)
MODEL_SERVER_PORT = int(MODEL_SERVER_PORT)

def select_best_model(valid_groupath, filters, trainer_name="", node_type=-1, weight=False):
    model_names = [f for f in os.listdir(valid_groupath) if \
                    f != CHECKPOINT_FOLDERNAME \
                    and not os.path.isfile(os.path.join(valid_groupath,f)) \
                    and (trainer_name == "" or trainer_name in f) \
                    and (node_type == -1 or str(node_type) in f) ]
    # Load metadata of trainers
    best_cadidate = None
    best_response = None
    for model_name in model_names:
        model_savepath = os.path.join(valid_groupath, model_name)
        metadata = load_json(model_savepath, METADATA_FILENAME)
        if metadata is None or not is_valid_model(metadata, filters) or ERROR_KEY not in metadata:
            # invalid metadata
            continue
        if weight:
            response = get_model_weight(valid_groupath, model_name)
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
@app.route('/model', methods=['POST'])
def get_model():
    model_request = request.get_json()
    req = ModelRequest(**model_request)
    # find valid feature groups from available metrics in request
    valid_fgs = get_valid_feature_groups(req.metrics)
    # parse filtering conditions of metadata attribute from request if exists  (e.g., minimum mae)
    filters = parse_filters(req.filter)
    ####### support old definition, TO-REMOVE #########
    if "Weight" in req.output_type:
        req.weight = True
    if "Dyn" in req.output_type:
        req.output_type = "DynPower"
    else:
        req.output_type = "AbsPower"
    ###################################################
    output_type = ModelOutputType[req.output_type]
    best_model = None
    best_response = None
    # find best model comparing best candidate from each valid feature group complied with filtering conditions
    for fg in valid_fgs:
        valid_groupath = get_model_group_path(output_type, fg, req.source)
        if os.path.exists(valid_groupath):
            best_candidate, response = select_best_model(valid_groupath, filters, req.trainer_name, req.node_type, req.weight)
            if best_candidate is None:
                continue
            if best_model is None or best_model[ERROR_KEY] > best_candidate[ERROR_KEY]:
                best_model = best_candidate
                best_response = response
    if best_model is None:
        return make_response("Cannot find model for {} at the moment".format(model_request), 400)
    if req.weight:
        try:
            response = app.response_class(
            response=json.dumps(best_response),
            status=200,
            mimetype='application/json'
            )
            return response
        except ValueError as err:
            return make_response("Get weight response error: {}".format(err), 400)
    else:
        try:
            return send_file(best_response, as_attachment=True)
        except ValueError as err:
            return make_response("Send archived model error: {}".format(err), 400)

# return name list of best-candidate pipelines
@app.route('/best-models', methods=['GET'])
def get_available_models():
    fg = request.args.get('fg')
    ot = request.args.get('ot')
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

        if filter is None:
            filters = dict()
        else: 
            filters = parse_filters(filter)

        model_names = dict()
        for output_type in output_types:
            model_names[output_type.name] = dict()
            for fg in valid_fgs:
                valid_groupath = get_model_group_path(output_type, fg)
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
        return make_response("Failed to get best model list: {}".format(err), 400)

if __name__ == '__main__':
   app.run(host="0.0.0.0", debug=True, port=MODEL_SERVER_PORT)
