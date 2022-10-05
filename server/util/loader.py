import os
import json

import requests
import codecs

FILTER_ITEM_DELIMIT = ';'
VALUE_DELIMIT = ':'
ARRAY_DELIMIT = ','

def load_json(path, json_file):
    filepath = os.path.join(path, json_file)
    try:
        with open(filepath) as f:
            res = json.load(f)
    except Exception as e:
        print(e)
        return None
    return res

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

def is_valid_model(metadata, filters):
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


def get_save_path(group_path, model_name):
    return os.path.join(group_path, model_name)

def get_archived_file(group_path, model_name):
    save_path = get_save_path(group_path, model_name)
    return save_path + '.zip'

# model file must be save in json
def get_model_weight(group_path, model_name, model_file):
    save_path = get_save_path(group_path, model_name)
    return load_json(save_path, model_file)

def download_and_save(url, filepath):
    try:
        response = requests.get(url)
    except Exception as e:
        print("Failed to load {} to {}: {}:".format(url, filepath, e))
        return None
    if response.status_code != 200:
        print("Failed to load {} to {}: {}:".format(url, filepath, response.status_code))
        return None
    with codecs.open(filepath, 'wb') as f:
        f.write(response.content)
    print("Successfully load {} to {}".format(url, filepath))
    return filepath