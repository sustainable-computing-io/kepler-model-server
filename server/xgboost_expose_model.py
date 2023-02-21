import json
from flask import Flask, request
from train.pipelines.XGBoostRegressionStandalonePipeline.pipe import XGBoostRegressionStandalonePipeline
from train.pipe_util import XGBoostRegressionTrainType

app = Flask(__name__)

MODEL_SERVER_PORT = 8100

@app.route('/xgboost', methods=['POST'])
def get_users():
    data = request.get_json(force=True)
    desired_prediction = json.loads(data['desired_prediction'])
    xgb_pipeline = XGBoostRegressionStandalonePipeline(XGBoostRegressionTrainType.KFoldCrossValidation, "XGBoost/")
    y_pred, model_desc = xgb_pipeline.predict([desired_prediction])
    model_desc["prediction"] = y_pred[0]
    response = app.response_class(
        response=json.dumps(model_desc),
        status=200,
        mimetype='application/json'
    )
    return response
    

if __name__ == '__main__':
   app.run(host="0.0.0.0", debug=True, port=MODEL_SERVER_PORT)
