# python clustering.py ../../resource/query_response
# ```mermaid
# graph TD;
# A[prom response json] -->|train.prom.prom_responses_to_results| B[kepler query result dict];
# B -->|profile.tool.profile_background.process| C[power profile dict\n source-component-node type to power];
# C -->|train.generate_profiles| D[Profile class dict\n node type to Profile]
# ```

import sys
import os

import json
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from datetime import datetime

import joblib

src_path = os.path.join(os.path.dirname(__file__), '..')
prom_path = os.path.join(os.path.dirname(__file__), '..', 'prom')
train_path = os.path.join(os.path.dirname(__file__), '..', 'train')

sys.path.append(src_path)
sys.path.append(prom_path)
sys.path.append(train_path)

from train.prom.prom_query import generate_dataframe_from_response, TIMESTAMP_COL
from train.extractor.extractor import node_info_query, cpu_frequency_info_query, energy_component_to_query
from train import PowerSourceMap

from train import DefaultProfiler
from train.profiler.generate_scaler import process as generate_scaler_process


clustering_model_filename = os.path.join(os.path.dirname(__file__), '..', '..', 'resource', 'kmeans.pkl')
number_of_cluster = 3

def _cr_filename(query_response_path, resultID):
    return os.path.join(query_response_path, resultID, "cr.yaml")

def _response_filename(query_response_path, resultID):
    return os.path.join(query_response_path, resultID, "response.json")

def get_benchmark_performance(query_response_path, resultID, scenario="t", config_wl="threads"):
    filename = _cr_filename(query_response_path, resultID)
    # Open the file and load the file
    with open(filename) as f:
        data = yaml.load(f, Loader=SafeLoader)
        results = data["status"]["results"]
        items = []
        for result in results:
            scene = result["scenarios"][scenario].split(";")[0]
            for rep in result["repetitions"]:
                item = rep.copy()
                item[config_wl] = scene
                items += [item]
    df = pd.DataFrame(items)
    df["performanceValue"] = df["performanceValue"].astype(float)
    df = df[[config_wl, "performanceValue"]].groupby(config_wl).median()
    return list(df.sort_values(by="performanceValue", ascending=False).values.squeeze())

def get_number_of_cores(cpu_frequency_df):
    return len(pd.unique(cpu_frequency_df["cpu"]))

def get_number_of_packages(package_power_df):
    return len(pd.unique(package_power_df["package"]))

def get_component_power_range(power_df, metric):
    return (power_df[metric].min(), power_df[metric].max(), round(power_df[metric].mean()), power_df[metric].sum())

def get_average_cpu_frequency(cpu_frequency_df):
    return cpu_frequency_df[cpu_frequency_info_query].astype(float).mean()

def get_cpu_architecture(response):
    node_info_df = generate_dataframe_from_response(node_info_query, response[node_info_query])
    cpu_architecture = pd.unique(node_info_df['cpu_architecture'])
    return list(cpu_architecture)

def save_model(model):
    with open(clustering_model_filename, "wb") as f:
        joblib.dump(model, f)

def load_model():
    try:
        with open(clustering_model_filename, "rb") as f:
            model = joblib.load(f)
            return model
    except:
        return None

def get_node_info(query_response_path, resultID):
    item = dict()
    response_filename = _response_filename(query_response_path, resultID)
    with open(response_filename, "r") as f:
        response = json.load(f)
        item["cpu_architecture"] = get_cpu_architecture(response)
        cpu_frequency_df = generate_dataframe_from_response(cpu_frequency_info_query, response[cpu_frequency_info_query])
        item["#core"] = get_number_of_cores(cpu_frequency_df)
        item["avg_freq"] = get_average_cpu_frequency(cpu_frequency_df)
        item["#pkg"] = 1
        for components in PowerSourceMap.values():
            for component in components:
                query = energy_component_to_query(component)
                if query not in response:
                    continue
                power_df = generate_dataframe_from_response(query, response[query])
                power_df[query] = power_df[query].astype(float).transform(round)
                if component == "package":
                    item["#pkg"] = get_number_of_packages(power_df)
                item[component +" (min,max,mean,sum)"] = get_component_power_range(power_df, query)
        return item, response

def response_to_result(response):
    results = dict()
    for query in response.keys():
        results[query] = generate_dataframe_from_response(query, response[query])
        if len(results[query]) > 0:
            if query == node_info_query:
                results[query][query] = results[query][query].astype(int)
            else:
                results[query][query] = results[query][query].astype(float)
    return results
    
def process(query_response_path):
    resultIDs = [ resultID for resultID in os.listdir(query_response_path)]
    node_infos = []
    x_values = []
    responses = dict()
    for resultID in resultIDs:
        performance_values = get_benchmark_performance(query_response_path, resultID)
        x_values += [performance_values]
        node_info, response = get_node_info(query_response_path, resultID)
        node_infos += [node_info]
        responses[resultID] = response
    X = np.array(x_values)
    kmeans = load_model()
    if kmeans is None:
        print("Create new clustering model")
        kmeans = MiniBatchKMeans(n_clusters=number_of_cluster, random_state=0)
    kmeans = kmeans.partial_fit(X)
    predicted_group = kmeans.predict(X)
    node_info_df = pd.DataFrame(node_infos)
    node_info_df["node_type"] = predicted_group
    node_info_df["node_type"] = node_info_df["node_type"].astype(int)
    node_info_df["performance_value"] = x_values
    print(node_info_df)
    node_info_df.to_csv(os.path.join(os.path.dirname(__file__),"clustered_data.csv"))
    save_model(kmeans)

    for index in range(len(resultIDs)):
        resultID = resultIDs[index]
        predicted_group_index = predicted_group[index]
        response = responses[resultID]
        result = response_to_result(response)
        # update node_type
        result[node_info_query][node_info_query] = int(predicted_group_index)
        generate_scaler_process(result)
        DefaultProfiler.process(result)

def find_mean(values):
    return np.array(values).mean()

if __name__ == "__main__":
    query_response_path = sys.argv[1]
    process(query_response_path)
