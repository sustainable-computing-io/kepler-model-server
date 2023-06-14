import os
import sys

src_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(src_path)

cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)

from model_server_connector import is_comp_output, ModelOutputType
from estimate_common import load_model_by_pickle, load_model_by_keras, load_model_by_json, transform_and_predict

class KerasModel():
    def __init__(self, model_path, model_name, output_type, model_file, features, fe_files, component_init=False):
        self.name = model_name
        self.features = features
        self.output_type = ModelOutputType[output_type]
        self.comp_type = not component_init and is_comp_output(self.output_type)
        if self.comp_type:
            self.models = dict()
            model_info = load_model_by_json(model_path, model_file)
            for comp, model_metadata in model_info.items():
                model = KerasModel(model_path, self.name, self.output_type.name, model_metadata['model_file'], model_metadata['features'], model_metadata['fe_files'], component_init=True)
                self.models[comp] = model
        else:
            self.model = load_model_by_keras(model_path, model_file)
            self.fe_list = []
            for fe_filename in fe_files:
                self.fe_list += [load_model_by_pickle(model_path, fe_filename)]

    def get_power(self, request):
        if self.comp_type:
            results = dict()
            for comp, model in self.models.items():
                 y, msg = transform_and_predict(model, request)
                 if msg != "":
                    return [], msg
                 results[comp] = y
            return results, msg
        else:
            return transform_and_predict(self, request)
