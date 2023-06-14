# offline trainer 
# input: response, pipeline attributes
# output: model output

# server:
#   python src/train/offline_trainer.py

# client (test):
#   python tests/offline_trainer_test.py <src_json_file> <save_path>

from train import NewPipeline, PowerSourceMap, FeatureGroup, load_class
from extractor import DefaultExtractor
from prom.prom_query import prom_responses_to_results

import os
util_path = os.path.join(os.path.dirname(__file__), 'util')
from util.loader import get_pipeline_path

import shutil

from flask import Flask, request, make_response, send_file
serve_port = 8080

"""
TrainRequest
name - pipeline/model name
output_type - target output
trainer - attribute to construct a training pipeline
    - abs_trainers - trainer class name list for absolute power training 
    - dyn_trainers - trainer class name list for dynamic power training
    - profiles - profiling data for profile-based idle power isolation
    - isolator - isolator class name
        - isolator_args - isolator arguments such as training class name to predict background power, profiled idle
data - data to train
response - prom_response from query.py

TrainResponse
zip file of pipeline folder or error message
"""

class TrainAttribute():
    def __init__(self, abs_trainers, dyn_trainers, profiles, isolator, isolator_args):
        self.abs_trainers = abs_trainers
        self.dyn_trainers = dyn_trainers
        self.profiles = profiles
        self.isolator = isolator
        self.isolator_args = isolator_args

class TrainRequest():
    def __init__(self, name, feature_group, power_source, trainer, data):
        self.name = name
        self.feature_group = FeatureGroup[feature_group]
        self.power_source = power_source
        self.energy_components = PowerSourceMap[self.power_source]
        self.trainer = TrainAttribute(**trainer)
        self.data = prom_responses_to_results(data)
        self.init_pipeline()
    
    def init_trainer(self, trainer_name, node_level):
        trainer_class = load_class("trainer", trainer_name)
        trainer = trainer_class(self.profiles, self.energy_components, self.feature_group.name, self.energy_source, node_level)
        return trainer
    
    def init_pipeline(self):
        isolator_class = load_class("isolator", self.trainer.isolator)
        isolator = isolator_class(self.trainer.isolator_args)
        self.pipeline = NewPipeline(self.name, self.profiles, self.abs_trainer_names, self.dyn_trainer_names, extractor=DefaultExtractor(), isolator=isolator)
    
    def get_model(self):
        # train model
        self.pipeline.process(self.data, self.energy_components,self.feature_group, self.energy_source)
        # return model
        pipeline_path = get_pipeline_path(pipeline_name=self.name)
        try:
            shutil.make_archive(pipeline_path, 'zip', pipeline_path)
            return pipeline_path + '.zip'
        except Exception as e:
            print(e)
            return None

app = Flask(__name__)

# return archive file or error
@app.route('/train', methods=['POST'])
def train():
    train_request = request.get_json()
    req = TrainRequest(**train_request)
    model = req.get_model()
    if model is None:
        return make_response("Cannot train model {}".format(req.name), 400)
    else:
        try:
            return send_file(model, as_attachment=True)
        except ValueError as err:
            return make_response("Send trained model error: {}".format(err), 400)

if __name__ == '__main__':
   app.run(host="0.0.0.0", debug=True, port=serve_port)
