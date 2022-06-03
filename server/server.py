from flask import Flask, redirect, url_for, request, json, current_app, send_from_directory
import os
app = Flask(__name__)

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

#Required to send models. There are no models created yet.
@app.route('/models/<model_type>', methods=['GET'])
def get_model(model_type='core'):
    models_directory = os.path.join(current_app.root_path, '/models/')
    if model_type == 'dram':
        return send_from_directory(directory=models_directory, filename='dram_model.h5')
    elif model_type == 'core':
        return send_from_directory(directory=models_directory, filename='core_model.h5')


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