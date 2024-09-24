# offline trainer
# input: response, pipeline attributes
# output: model output

# server:
#   python src/train/offline_trainer.py

# client (test):
#   python tests/offline_trainer_test.py <src_json_file> <save_path>

import importlib
import shutil

from flask import Flask, make_response, request, send_file

from kepler_model.train.extractor.extractor import DefaultExtractor
from kepler_model.train.isolator.isolator import ProfileBackgroundIsolator
from kepler_model.train.isolator.train_isolator import TrainIsolator
from kepler_model.train.pipeline import NewPipeline
from kepler_model.train.profiler.profiler import Profiler, generate_profiles
from kepler_model.util.config import model_toppath
from kepler_model.util.loader import default_pipelines, get_pipeline_path
from kepler_model.util.prom_types import get_valid_feature_group_from_queries, prom_responses_to_results
from kepler_model.util.train_types import PowerSourceMap

serve_port = 8102

"""
TrainRequest
name - pipeline/model name
energy_source - target enery source to train for (such as rapl-sysfs, acpi) 
trainer - attribute to construct a training pipeline
    - abs_trainers - trainer class name list for absolute power training 
    - dyn_trainers - trainer class name list for dynamic power training
    - idle_prom_response - prom response at idle state for profile-based idle power isolation 
    - isolator - isolator key name  
        - isolator_args - isolator arguments such as training class name to predict background power, profiled idle 
prome_response - prom response with workload for power model training

TrainResponse
zip file of pipeline folder or error message
"""


class TrainAttribute:
    def __init__(self, abs_trainers, dyn_trainers, idle_prom_response, isolator, isolator_args):
        self.abs_trainers = abs_trainers
        self.dyn_trainers = dyn_trainers
        self.idle_prom_response = idle_prom_response
        self.isolator = isolator
        self.isolator_args = isolator_args


class TrainRequest:
    def __init__(self, name, energy_source, trainer, prom_response, use_vm_metrics=False):
        self.name = name
        self.energy_source = energy_source
        self.use_vm_metrics = use_vm_metrics
        if trainer is not None:
            self.trainer = TrainAttribute(**trainer)
        self.prom_response = prom_response

    def init_isolator(self, profiler, profiles, idle_data):
        isolator_key = self.trainer.isolator
        isolator_args = self.trainer.isolator_args
        print(isolator_key)
        if isolator_key == ProfileBackgroundIsolator.__name__:
            isolator = ProfileBackgroundIsolator(profiles, idle_data)
        elif isolator_key == TrainIsolator.__name__:
            if "abs_pipeline_name" not in isolator_args:
                # use default pipeline for absolute model training in isolation
                isolator_args["abs_pipeline_name"] = default_pipelines[self.energy_source]
            isolator = TrainIsolator(idle_data, profiler=profiler, abs_pipeline_name=isolator_args["abs_pipeline_name"])
        else:
            module_path = importlib.import_module("kepler_model.train.isolator.isolator")
            # default init, no args
            isolator = getattr(module_path, isolator_key)()
        return isolator

    def init_pipeline(self):
        profiler = Profiler(extractor=DefaultExtractor())
        # TODO: if idle_data is None use profile from registry
        idle_data = prom_responses_to_results(self.trainer.idle_prom_response)
        idle_profile_map = profiler.process(idle_data)
        profiles = generate_profiles(idle_profile_map)
        isolator = self.init_isolator(profiler, profiles, idle_data)
        valid_feature_groups = get_valid_feature_group_from_queries(idle_data.keys())
        self.pipeline = NewPipeline(
            self.name,
            self.trainer.abs_trainers,
            self.trainer.dyn_trainers,
            extractor=DefaultExtractor(),
            isolator=isolator,
            target_energy_sources=[self.energy_source],
            valid_feature_groups=valid_feature_groups,
            use_vm_metrics=self.use_vm_metrics,
        )

    def get_model(self):
        self.init_pipeline()
        energy_components = PowerSourceMap[self.energy_source]
        # train model
        data = prom_responses_to_results(self.prom_response)
        valid_feature_groups = get_valid_feature_group_from_queries(data.keys())
        for feature_group in valid_feature_groups:
            self.pipeline.process(data, energy_components, self.energy_source, feature_group.name)
        # return model
        pipeline_path = get_pipeline_path(model_toppath=model_toppath, pipeline_name=self.name)
        self.pipeline.save_metadata()
        try:
            shutil.make_archive(pipeline_path, "zip", pipeline_path)
            return pipeline_path + ".zip"
        except Exception as e:
            print(e)
            return None


app = Flask(__name__)


# return archive file or error
@app.route("/train", methods=["POST"])
def train():
    train_request = request.get_json()
    req = TrainRequest(**train_request)
    model = req.get_model()
    print(f"Get Model: {model}")
    if model is None:
        return make_response(f"Cannot train model {req.name}", 400)
    else:
        try:
            return send_file(model, as_attachment=True)
        except ValueError as err:
            return make_response(f"Send trained model error: {err}", 400)


def run():
    app.run(host="0.0.0.0", port=serve_port)
