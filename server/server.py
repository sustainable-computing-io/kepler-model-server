from ast import mod
from multiprocessing.sharedctypes import Value
from flask import Flask, redirect, url_for, request, json, current_app, send_from_directory, make_response
import os
from kepler_model_trainer import archive_saved_model, return_model_weights

app = Flask(__name__)

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
        model_weights_dict = return_model_weights(model_type)
        response = app.response_class(
        response=json.dumps(model_weights_dict),
        status=200,
        mimetype='application/json'
        )
        return response
    except ValueError:
        return make_response("Model '" + model_type + "' is not valid", 400)
    except FileNotFoundError:
        return make_response("Model '" + model_type + "' does not exist at the moment", 400)
    
if __name__ == '__main__':
   app.run(host="0.0.0.0", debug=True, port=8100)
   