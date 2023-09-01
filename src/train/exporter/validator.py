import os
import sys

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

from loader import load_train_args
from config import ERROR_KEY
from train_types import PowerSourceMap

required_benchmark = ["stressng_kepler_query"]

default_threshold_percent = 20

def validate_arguments(pipeline_path):
    train_args = load_train_args(pipeline_path)
    inputs = train_args["input"].split(",")
    missing_inputs = [input for input in required_benchmark if input not in inputs]
    if len(missing_inputs) > 0:
        print("missing required training inputs: ", missing_inputs)
        return False
    return True
    
def find_acceptable_mae(preprocess_data, metadata):
    power_cols = [col for col in preprocess_data.columns if "power" in col]
    power_range = preprocess_data[power_cols].max() - preprocess_data[power_cols].min()
    mae_threshold = default_threshold_percent * power_range.max() / 100
    return metadata[ERROR_KEY] <= mae_threshold, power_range.max(), mae_threshold