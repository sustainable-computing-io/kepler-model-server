from ast import mod
from multiprocessing.sharedctypes import Value
from flask import Flask, redirect, url_for, request, json, current_app, send_from_directory, make_response
import os
from kepler_model_trainer import archive_saved_model, return_model_weights

app = Flask(__name__)

# Obsolete Test Legacy Code: June 13, 2022
@app.route('/model',methods = ['POST', 'GET'])
def model():
   if request.method == 'GET':
      ret = makeCoeff()
      app.logger.warn("BODY: %s" % ret)
      return ret

# Obsolete Test Legacy Code: June 13, 2022
@app.route('/data',methods = ['POST', 'GET'])
def data():
   if request.method == 'POST':
      app.logger.warn("BODY: %s" % request.get_data())
      return 'success'


# Acceptable model_type values: core_model, dram_model

# Returns a trained model given the model type/name. Currently, the route will return an archived file of keras's SavedModel.
# The extracted archived SavedModel can be directly fitted. To download the SavedModel as attachment, include as_attachment=True in the 
# send_from_directory flask function. Other lightweight options include sending the model as a HDF5 format, sending the model
# structure as a JSON and the model's weights as an HDF5 file. 
@app.route('/models/<model_type>', methods=['GET'])
@app.route('/models/')
def get_model(model_type='core_model'):
    #models_directory = os.path.join(current_app.root_path, 'models/')
    try:
        filepath, name_of_file = archive_saved_model(model_type)
        print(name_of_file)
        return send_from_directory(filepath, name_of_file, as_attachment=True)
    except ValueError:
        return make_response("Model '" + model_type + "' is not valid", 400)
    except FileNotFoundError:
        return make_response("Model '" + model_type + "' does not exist at the moment", 400)

    #models_directory = os.path.join(current_app.root, 'models/')
    #if model_type == 'dram_model' and os.path.exists(os.path.join(models_directory, 'dram_model')): # must check whether the model was created or not
         #shutil.make_archive(os.path.join(models_directory, 'dram_model'), 'zip', os.path.join(models_directory, 'dram_model')) # this function should overrite an existing zipped model
    #    return send_from_directory(models_directory, 'dram_model.zip')
    #if model_type == 'core_model' and os.path.exists(os.path.join(models_directory, 'core_model')): # must check whether the model was created or not
    #    shutil.make_archive(os.path.join(models_directory, 'core_model'), 'zip', os.path.join(models_directory, 'core_model')) # this function should overrite an existing zipped model
    #    return send_from_directory(models_directory, 'core_model.zip')
    #return make_response("Model '" + model_type + "' does not exist at the moment", 400)

# Returns coefficients/weights of the trained regression model.
@app.route('/model-weights/<model_type>', methods=['GET'])
@app.route('/model-weights/')
def get_model_weights(model_type='core_model'):
    try:
        #returned_coefficients = return_model_coefficients(model_type)
        kernel_matrix, bias = return_model_weights(model_type)
        #print(kernel_matrix)
        #print(bias)
        data = {
            "kernel_matrix": kernel_matrix,
            "bias": bias
        }
        response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
        )
        return response
    except ValueError:
        return make_response("Model '" + model_type + "' is not valid", 400)
    except FileNotFoundError:
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
   