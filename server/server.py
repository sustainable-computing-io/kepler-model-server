from ast import mod
from multiprocessing.sharedctypes import Value
from flask import Flask, redirect, url_for, request, json, current_app, send_from_directory, make_response, Response
from kepler_model_trainer import archive_saved_model, return_model_weights
from prometheus_client import generate_latest, Gauge


app = Flask(__name__)
# Acceptable model_type values: core_model, dram_model
#Prometheus Weight Metrics
prometheus_all_model_weights_gauge_dict = {'curr_cpu_cycles_core_model_weight': Gauge('curr_cpu_cycles_core_model_weight', 'corresponding coefficient for curr_cpu_cycles label for core model'),
                                          'curr_cpu_cycles_core_model_mean': Gauge('curr_cpu_cycles_core_model_mean', 'corresponding mean for curr_cpu_cycles label for core model'),
                                          'curr_cpu_cycles_core_model_variance': Gauge('curr_cpu_cycles_core_model_variance', 'corresponding variance for curr_cpu_cycles label for core model'),

                                          'current_cpu_instructions_core_model_weight': Gauge('current_cpu_instructions_core_model_weight', 'corresponding coefficient for current_cpu_instructions label for core model'),
                                          'current_cpu_instructions_core_model_mean': Gauge('current_cpu_instructions_core_model_mean', 'corresponding mean for current_cpu_instructions label for core model'),
                                          'current_cpu_instructions_core_model_variance': Gauge('current_cpu_instructions_core_model_variance', 'corresponding variance for current_cpu_instructions label for core model'),

                                          'curr_cpu_time_core_model_weight': Gauge('curr_cpu_time_core_model_weight', 'corresponding coefficient for curr_cpu_time label for core model'),
                                          'curr_cpu_time_core_model_mean': Gauge('curr_cpu_time_core_model_mean', 'corresponding mean for curr_cpu_time label for core model'),
                                          'curr_cpu_time_core_model_variance': Gauge('curr_cpu_time_core_model_variance', 'corresponding variance for curr_cpu_time label for core model'),

                                          'cpu_architecture_sandy_bridge_core_model_weight': Gauge('cpu_architecture_sandy_bridge_core_model_weight', 'corresponding coefficient for cpu_architecture_sandy_bridge label for core model'),
                                          'cpu_architecture_ivy_bridge_core_model_weight': Gauge('cpu_architecture_ivy_bridge_core_model_weight', 'corresponding coefficient for cpu_architecture_ivy_bridge label for core model'),
                                          'cpu_architecture_haswell_core_model_weight': Gauge('cpu_architecture_haswell_core_model_weight', 'corresponding coefficient for cpu_architecture_haswell label for core model'),
                                          'cpu_architecture_broadwell_core_model_weight': Gauge('cpu_architecture_broadwell_core_model_weight', 'corresponding coefficient for cpu_architecture_broadwell label for core model'),
                                          'cpu_architecture_sky_lake_core_model_weight': Gauge('cpu_architecture_sky_lake_core_model_weight', 'corresponding coefficient for cpu_architecture_sky_lake label for core model'),
                                          'cpu_architecture_cascade_lake_core_model_weight': Gauge('cpu_architecture_cascade_lake_core_model_weight', 'corresponding coefficient for cpu_architecture_cascade_lake label for core model'),
                                          'cpu_architecture_coffee_lake_core_model_weight': Gauge('cpu_architecture_coffee_lake_core_model_weight', 'corresponding coefficient for cpu_architecture_coffee_lake label for core model'),
                                          'cpu_architecture_alder_lake_core_model_weight': Gauge('cpu_architecture_alder_lake_core_model_weight', 'corresponding coefficient for cpu_architecture_alder_lake label for core model'),
                                          'core_model_bias_weight': Gauge('core_model_bias_weight', 'corresponding coefficient for bias label for core model'),

                                          'curr_resident_memory_dram_model_weight': Gauge('curr_resident_memory_dram_model_weight', 'corresponding coefficient for curr_resident_memory label for dram model'),
                                          'curr_resident_memory_dram_model_mean': Gauge('curr_resident_memory_dram_model_mean', 'corresponding mean for curr_resident_memory label for dram model'),
                                          'curr_resident_memory_dram_model_variance': Gauge('curr_resident_memory_dram_model_variance', 'corresponding variance for curr_resident_memory label for dram model'),

                                          'curr_cache_misses_dram_model_weight': Gauge('curr_cache_misses_dram_model_weight', 'corresponding coefficient for curr_cache_misses label for dram model'),
                                          'curr_cache_misses_dram_model_mean': Gauge('curr_cache_misses_dram_model_mean', 'corresponding mean for curr_cache_misses label for dram model'),
                                          'curr_cache_misses_dram_model_variance': Gauge('curr_cache_misses_dram_model_variance', 'corresponding variance for curr_cache_misses label for dram model'),

                                          'cpu_architecture_sandy_bridge_dram_model_weight': Gauge('cpu_architecture_sandy_bridge_dram_model_weight', 'corresponding coefficient for cpu_architecture_sandy_bridge label for dram model'),
                                          'cpu_architecture_ivy_bridge_dram_model_weight': Gauge('cpu_architecture_ivy_bridge_dram_model_weight', 'corresponding coefficient for cpu_architecture_ivy_bridge label for dram model'),
                                          'cpu_architecture_haswell_dram_model_weight': Gauge('cpu_architecture_haswell_dram_model_weight', 'corresponding coefficient for cpu_architecture_haswell label for dram model'),
                                          'cpu_architecture_broadwell_dram_model_weight': Gauge('cpu_architecture_broadwell_dram_model_weight', 'corresponding coefficient for cpu_architecture_broadwell label for dram model'),
                                          'cpu_architecture_sky_lake_dram_model_weight': Gauge('cpu_architecture_sky_lake_dram_model_weight', 'corresponding coefficient for cpu_architecture_sky_lake label for dram model'),
                                          'cpu_architecture_cascade_lake_dram_model_weight': Gauge('cpu_architecture_cascade_lake_dram_model_weight', 'corresponding coefficient for cpu_architecture_cascade_lake label for dram model'),
                                          'cpu_architecture_coffee_lake_dram_model_weight': Gauge('cpu_architecture_coffee_lake_dram_model_weight', 'corresponding coefficient for cpu_architecture_coffee_lake label for dram model'),
                                          'cpu_architecture_alder_lake_dram_model_weight': Gauge('cpu_architecture_alder_lake_dram_model_weight', 'corresponding coefficient for cpu_architecture_alder_lake label for dram model'),
                                          'dram_model_bias_weight': Gauge('dram_model_bias_weight', 'corresponding coefficient for bias label for dram model')
                                          }

def update_prometheus_weights_for_given_model(model_type):
    new_model_weights = return_model_weights(model_type)
    if model_type == "core_model":
        numerical_variables_coefficients = new_model_weights['All_Weights']['Numerical_Variables']
        print(numerical_variables_coefficients['curr_cpu_cycles'])
        prometheus_all_model_weights_gauge_dict['curr_cpu_cycles_core_model_weight'].set(numerical_variables_coefficients['curr_cpu_cycles']['weight'])
        prometheus_all_model_weights_gauge_dict['curr_cpu_cycles_core_model_mean'].set(numerical_variables_coefficients['curr_cpu_cycles']['mean'])
        prometheus_all_model_weights_gauge_dict['curr_cpu_cycles_core_model_variance'].set(numerical_variables_coefficients['curr_cpu_cycles']['variance'])

        prometheus_all_model_weights_gauge_dict['current_cpu_instructions_core_model_weight'].set(numerical_variables_coefficients['current_cpu_instructions']['weight'])
        prometheus_all_model_weights_gauge_dict['current_cpu_instructions_core_model_mean'].set(numerical_variables_coefficients['current_cpu_instructions']['mean'])
        prometheus_all_model_weights_gauge_dict['current_cpu_instructions_core_model_variance'].set(numerical_variables_coefficients['current_cpu_instructions']['variance'])

        prometheus_all_model_weights_gauge_dict['curr_cpu_time_core_model_weight'].set(numerical_variables_coefficients['curr_cpu_time']['weight'])
        prometheus_all_model_weights_gauge_dict['curr_cpu_time_core_model_mean'].set(numerical_variables_coefficients['curr_cpu_time']['mean'])
        prometheus_all_model_weights_gauge_dict['curr_cpu_time_core_model_variance'].set(numerical_variables_coefficients['curr_cpu_time']['variance'])

        categorical_variables_cpu_architecture_coefficients = new_model_weights['All_Weights']['Categorical_Variables']['cpu_architecture']
        prometheus_all_model_weights_gauge_dict['cpu_architecture_sandy_bridge_core_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Sandy Bridge']['weight'])
        prometheus_all_model_weights_gauge_dict['cpu_architecture_ivy_bridge_core_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Ivy Bridge']['weight'])
        prometheus_all_model_weights_gauge_dict['cpu_architecture_haswell_core_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Haswell']['weight'])
        prometheus_all_model_weights_gauge_dict['cpu_architecture_broadwell_core_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Broadwell']['weight'])
        prometheus_all_model_weights_gauge_dict['cpu_architecture_sky_lake_core_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Sky Lake']['weight']) 
        prometheus_all_model_weights_gauge_dict['cpu_architecture_cascade_lake_core_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Cascade Lake']['weight'])
        prometheus_all_model_weights_gauge_dict['cpu_architecture_coffee_lake_core_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Coffee Lake']['weight'])
        prometheus_all_model_weights_gauge_dict['cpu_architecture_alder_lake_core_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Alder Lake']['weight'])

        prometheus_all_model_weights_gauge_dict['core_model_bias_weight'].set(new_model_weights['All_Weights']['Bias_Weight'])
    elif model_type == "dram_model":
        numerical_variables_coefficients = new_model_weights['All_Weights']['Numerical_Variables']
        prometheus_all_model_weights_gauge_dict['curr_resident_memory_dram_model_weight'].set(numerical_variables_coefficients['curr_resident_memory']['weight'])
        prometheus_all_model_weights_gauge_dict['curr_resident_memory_dram_model_mean'].set(numerical_variables_coefficients['curr_resident_memory']['mean'])
        prometheus_all_model_weights_gauge_dict['curr_resident_memory_dram_model_variance'].set(numerical_variables_coefficients['curr_resident_memory']['variance'])

        prometheus_all_model_weights_gauge_dict['curr_cache_misses_dram_model_weight'].set(numerical_variables_coefficients['curr_cache_misses']['weight'])
        prometheus_all_model_weights_gauge_dict['curr_cache_misses_dram_model_mean'].set(numerical_variables_coefficients['curr_cache_misses']['mean'])
        prometheus_all_model_weights_gauge_dict['curr_cache_misses_dram_model_variance'].set(numerical_variables_coefficients['curr_cache_misses']['variance'])

        categorical_variables_cpu_architecture_coefficients = new_model_weights['All_Weights']['Categorical_Variables']['cpu_architecture']
        prometheus_all_model_weights_gauge_dict['cpu_architecture_sandy_bridge_dram_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Sandy Bridge']['weight'])
        prometheus_all_model_weights_gauge_dict['cpu_architecture_ivy_bridge_dram_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Ivy Bridge']['weight'])
        prometheus_all_model_weights_gauge_dict['cpu_architecture_haswell_dram_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Haswell']['weight'])
        prometheus_all_model_weights_gauge_dict['cpu_architecture_broadwell_dram_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Broadwell']['weight'])
        prometheus_all_model_weights_gauge_dict['cpu_architecture_sky_lake_dram_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Sky Lake']['weight']) 
        prometheus_all_model_weights_gauge_dict['cpu_architecture_cascade_lake_dram_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Cascade Lake']['weight'])
        prometheus_all_model_weights_gauge_dict['cpu_architecture_coffee_lake_dram_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Coffee Lake']['weight'])
        prometheus_all_model_weights_gauge_dict['cpu_architecture_alder_lake_dram_model_weight'].set(categorical_variables_cpu_architecture_coefficients['Alder Lake']['weight'])

        prometheus_all_model_weights_gauge_dict['dram_model_bias_weight'].set(new_model_weights['All_Weights']['Bias_Weight'])

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

@app.route('/metrics')
def get_metrics():
    try:
        update_prometheus_weights_for_given_model('core_model')
        update_prometheus_weights_for_given_model('dram_model')
        res = []
        for model_gauge in prometheus_all_model_weights_gauge_dict.values():
            res.append(generate_latest(model_gauge))
        
        return Response(res, mimetype='text/plain')
    except Exception as e:
        return make_response(str(e), 400)
    

if __name__ == '__main__':
   app.run(host="0.0.0.0", debug=True, port=8100)
   