import os
import sys
import logging

fpath = os.path.join(os.path.dirname(__file__), 'model')
sys.path.append(fpath)

util_path = os.path.join(os.path.dirname(__file__), '..', 'util')
sys.path.append(util_path)

from model_server_connector import unpack, ModelOutputType
from config import get_init_model
from loader import load_metadata

failed_list = []

FILTER_ITEM_DELIMIT = ';'
VALUE_DELIMIT = ':'
ARRAY_DELIMIT = ','

import requests

def parse_filters(filter):
    filter_list = filter.split(FILTER_ITEM_DELIMIT)
    filters = dict()
    for filter_item in filter_list:
        splits = filter_item.split(VALUE_DELIMIT)
        if len(splits) != 2:
            continue
        key = splits[0]
        if key == 'features':
            value = splits[1].split(ARRAY_DELIMIT)
        else:
            value = splits[1]
        filters[key] = value
    return filters 


def valid_metrics(metrics, features):
    for feature in features:
        if feature not in metrics:
            return False
    return True

def is_valid_model(metrics, metadata, filters):
    if not valid_metrics(metrics, metadata['features']):
        return False
    for attrb, val in filters.items():
        if not hasattr(metadata, attrb) or getattr(metadata, attrb) is None:
            print('{} has no {}'.format(metadata['model_name'], attrb))
            return False
        else:
            cmp_val = getattr(metadata, attrb)
            val = float(val)
            if attrb == 'abs_max_corr': # higher is better
                valid = cmp_val >= val
            else: # lower is better
                valid = cmp_val <= val
            if not valid:
                return False
    return True

def reset_failed_list():
     global failed_list
     failed_list = []

def get_achived_model(power_request):  
    print("get archived model")
    global failed_list
    output_type_name = power_request.output_type
    if output_type_name in failed_list:
        return None
    output_type = ModelOutputType[power_request.output_type]
    url = get_init_model(output_type_name)
    if url == "":
        print("No URL set for ", output_type_name)
        return None
    print("Try getting archieved model from URL: {} for {}".format(url, output_type_name))
    response = requests.get(url)
    print(response)
    if response.status_code != 200:
        return None
    output_path = unpack(output_type, response, replace=False)
    if output_path is not None:
        metadata = load_metadata(output_path)
        filters = parse_filters(power_request.filter)
        if not is_valid_model(power_request.metrics, metadata, filters):
            failed_list += [output_type_name]
            return None
    return output_path