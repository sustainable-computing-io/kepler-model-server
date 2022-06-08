from ast import mod
from flask import Flask, redirect, url_for, request, json, current_app, send_from_directory, make_response
import os
app = Flask(__name__)
import shutil


@app.route('/model',methods = ['POST', 'GET'])
def model():
   if request.method == 'GET':
      ret = makeCoeff()
      app.logger.warn("BODY: %s" % ret)
      return ret


@app.route('/data',methods = ['POST', 'GET'])
def data():
   if request.method == 'POST':
      app.logger.warn("BODY: %s" % request.get_data())
      return 'success'


# Returns a trained model given the model type/name. Currently, the route will return an archived file of keras's SavedModel.
# The extracted archived SavedModel can be directly fitted. To download the SavedModel as attachment, include as_attachment=True in the 
# send_from_directory flask function. Other lightweight options include sending the model as a HDF5 format, sending the model
# structure as a JSON and the model's weights as an HDF5 file. 
@app.route('/models/<model_type>', methods=['GET'])
@app.route('/models/')
def get_model(model_type='core_model'):
    models_directory = os.path.join(current_app.root_path, 'models/')
    if model_type == 'dram_model' and os.path.exists(os.path.join(models_directory, 'dram_model')): # must check whether the model was created or not
        shutil.make_archive(os.path.join(models_directory, 'dram_model'), 'zip', os.path.join(models_directory, 'dram_model')) # this function should overrite an existing zipped model
        return send_from_directory(models_directory, 'dram_model.zip')
    elif model_type == 'core_model' and os.path.exists(os.path.join(models_directory, 'core_model')): # must check whether the model was created or not
        shutil.make_archive(os.path.join(models_directory, 'core_model'), 'zip', os.path.join(models_directory, 'core_model')) # this function should overrite an existing zipped model
        return send_from_directory(models_directory, 'core_model.zip') 
    else:
        return make_response("Model '" + model_type + "' does not exist at the moment", 400)


def makeCoeff():
    data = {
      "cpu_time": 0.6,
      "cpu_cycles": 0.2,
      "cpu_instructions": 0.2,
      "memory_usage": 0.5,
      "cache_misses": 0.5,
    }
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
   app.run(host="0.0.0.0", debug=True, port=8100)
   