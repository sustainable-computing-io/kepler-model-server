from flask import Flask, request, json, make_response, send_file

import os
import sys
util_path = os.path.join(os.path.dirname(__file__), 'util')
train_path = os.path.join(os.path.dirname(__file__), 'train')

sys.path.append(util_path)
sys.path.append(train_path)

from train.pipeline import get_model_group_path, METADATA_FILENAME
from train.train_types import get_valid_feature_groups, is_weight_output, ModelOutputType
from util.config import getConfig
from util.loader import parse_filters, is_valid_model, load_json, get_model_weight, get_archived_file

###############################################
# model request 

class ModelRequest():
    def __init__(self, metrics, output_type, model_name="", filter=""):
        self.metrics = metrics
        self.model_name = model_name
        self.filter = filter
        self.output_type = output_type

###########################################

ERROR_KEY = 'mae'
ERROR_KEY = getConfig('ERROR_KEY', ERROR_KEY)
MODEL_SERVER_PORT = 8100
MODEL_SERVER_PORT = getConfig('MODEL_SERVER_PORT', MODEL_SERVER_PORT)
MODEL_SERVER_PORT = int(MODEL_SERVER_PORT)

def select_best_model(output_type, valid_groupath, filters):
    model_names = [f for f in os.listdir(valid_groupath) if not os.path.isfile(os.path.join(valid_groupath,f))]
    print("Load metadata of models:", model_names)
    best_cadidate = None
    best_response = None
    for model_name in model_names:
        model_savepath = os.path.join(valid_groupath, model_name)
        metadata = load_json(model_savepath, METADATA_FILENAME)
        if metadata is None or not is_valid_model(metadata, filters) or ERROR_KEY not in metadata:
            print("invalid metadata", metadata)
            continue
        if is_weight_output(output_type):
            response = get_model_weight(valid_groupath, model_name, metadata['model_file'])
            if response is None:
                # fail to get weight file
                print("cannot get model weight", response)
                continue
        else:
            response = get_archived_file(valid_groupath, model_name)
            if not os.path.exists(response):
                # archived model file does not exists
                print("cannot get archived_file", response)
                continue
        if best_cadidate is None or best_cadidate[ERROR_KEY] > metadata[ERROR_KEY]:
            best_cadidate = metadata
            best_response = response
    return best_cadidate, best_response

app = Flask(__name__)

@app.route('/model', methods=['POST'])
def get_model():
    model_request = request.get_json()
    req = ModelRequest(**model_request)
    valid_fgs = get_valid_feature_groups(req.metrics)
    filters = parse_filters(req.filter)
    print(req.output_type)
    output_type = ModelOutputType[req.output_type]
    best_model = None
    best_response = None
    print("valid feature groups: ", valid_fgs)
    for fg in valid_fgs:
        valid_groupath = get_model_group_path(output_type, fg)
        if os.path.exists(valid_groupath):
            best_candidate, response = select_best_model(output_type, valid_groupath, filters)
            if best_candidate is None:
                continue
            if best_model is None or best_model[ERROR_KEY] > best_candidate[ERROR_KEY]:
                best_model = best_candidate
                best_response = response
    if best_model is None:
        return make_response("Cannot find model for {} at the moment".format(model_request), 400)
    print(best_response)
    if is_weight_output(output_type):
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

if __name__ == '__main__':
   app.run(host="0.0.0.0", debug=True, port=MODEL_SERVER_PORT)
