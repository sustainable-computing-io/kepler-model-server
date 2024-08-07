############################################################
##
## generate_scaler
## generate a scaler for each node type from prom query
## 
## ./python generate_scaler.py query_output_folder
## e.g., ./python generate_scaler.py ../tests/data/prom_output
##
## input must be a query output of loaded state 
##
############################################################

import  sys
import  os

src_path = os.path.join(os.path.dirname(__file__), '..', '..')
prom_path = os.path.join(os.path.dirname(__file__), '..', '..', 'prom')
train_path = os.path.join(os.path.dirname(__file__), '..', '..', 'train')
util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
tool_path = os.path.join(os.path.dirname(__file__), '..', '..', 'profile', 'tool')

sys.path.append(src_path)
sys.path.append(prom_path)
sys.path.append(train_path)
sys.path.append(util_path)
sys.path.append(tool_path)

from sklearn.preprocessing import  MaxAbsScaler

from train import  DefaultExtractor, node_info_column, FeatureGroups, FeatureGroup, TIMESTAMP_COL
from util.train_types import  SYSTEM_FEATURES
from profile_background import  profile_path

import  pandas as pd
import  pickle

extractor = DefaultExtractor()

max_scaler_top_path = os.path.join(profile_path, '..', 'max_scaler')

if not os.path.exists(max_scaler_top_path):
    os.mkdir(max_scaler_top_path)

def read_query_results(query_path):
    results = dict()
    metric_filenames = [ metric_filename for metric_filename in os.listdir(query_path) ]
    for metric_filename in metric_filenames:
        metric = metric_filename.replace(".csv", "")
        filepath = os.path.join(query_path, metric_filename)
        results[metric] = pd.read_csv(filepath)
    return results

def save_scaler(scaler, node_type, feature_group, scaler_top_path):
    node_type_path = os.path.join(scaler_top_path, str(node_type))
    if not os.path.exists(node_type_path):
        os.mkdir(node_type_path)
    filename = os.path.join(node_type_path, feature_group + ".pkl")
    with open(filename, "wb") as f:
        pickle.dump(scaler, f)

def process(query_results):
    node_info_data = extractor.get_system_category(query_results)
    if node_info_data is None:
        print("No Node Info")
        return None
    node_types = pd.unique(node_info_data[node_info_column])
    for node_type in node_types:
        for feature_group in FeatureGroups:
            feature_group_name = feature_group.name
            features = FeatureGroups[FeatureGroup[feature_group_name]]
            workload_features = [feature for feature in features if feature not in SYSTEM_FEATURES]
            system_features = [feature for feature in features if feature in SYSTEM_FEATURES]
            feature_data = extractor.get_workload_feature_data(query_results, workload_features)
            if feature_data is None:
                print("cannot process ", feature_group_name)
                continue
            workload_feature_data = feature_data.groupby([TIMESTAMP_COL]).sum()[workload_features]
            if len(system_features) > 0:
                system_feature_data = extractor.get_system_feature_data(query_results, system_features)
                feature_data = workload_feature_data.join(system_feature_data).sort_index().dropna()
            else: 
                feature_data = workload_feature_data

            feature_data = feature_data.join(node_info_data)
            node_types = pd.unique(feature_data[node_info_column])
            # filter and extract features
            x_values = feature_data[feature_data[node_info_column]==node_type][features].values
            max_scaler = MaxAbsScaler()
            max_scaler.fit(x_values)
            save_scaler(max_scaler, node_type, feature_group_name, max_scaler_top_path)
