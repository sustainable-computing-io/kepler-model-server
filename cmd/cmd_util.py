import os
import sys
import datetime
import pandas as pd

UTC_OFFSET_TIMEDELTA = datetime.datetime.utcnow() - datetime.datetime.now()

cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)

from util.prom_types import node_info_column, prom_responses_to_results
from util.train_types import ModelOutputType, FeatureGroup
from util.loader import load_json, get_pipeline_path
from util.saver import assure_path, save_csv

def print_file_to_stdout(data_path, args):
    file_path = os.path.join(data_path, args.output)
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
            print(contents)
    except FileNotFoundError:
        print(f"Error: Output '{file_path}' not found.")
    except IOError:
        print(f"Error: Unable to read output '{file_path}'.")

def extract_time(data_path, benchmark_filename):
    data = load_json(data_path, benchmark_filename)
    if benchmark_filename != "customBenchmark":
        start_str = data["metadata"]["creationTimestamp"]
        start = datetime.datetime.strptime(start_str, '%Y-%m-%dT%H:%M:%SZ')
        end_str = data["status"]["results"][-1]["repetitions"][-1]["pushedTime"].split(".")[0]
        end = datetime.datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
    else:
        start_str = data["startTimeUTC"]
        start = datetime.datetime.strptime(start_str, '%Y-%m-%dT%H:%M:%SZ')
        end_str = data["endTimeUTC"]
        end = datetime.datetime.strptime(end_str, '%Y-%m-%dT%H:%M:%SZ')
    print(UTC_OFFSET_TIMEDELTA)
    return start-UTC_OFFSET_TIMEDELTA, end-UTC_OFFSET_TIMEDELTA

def save_query_results(data_path, output_filename, query_response):
    query_results = prom_responses_to_results(query_response)
    save_path = os.path.join(data_path, "{}_csv".format(output_filename))
    assure_path(save_path)
    for query, data in query_results.items():
        save_csv(save_path, query, data)

def summary_validation(validate_df):
    if len(validate_df) == 0:
        print("No data for validation.")
        return
    items = []
    metric_to_validate_pod = {
        "cgroup": "kepler_container_cgroupfs_cpu_usage_us_total",
        # "hwc": "kepler_container_cpu_instructions_total",
        "hwc": "kepler_container_cpu_cycles_total",
        "kubelet": "kepler_container_kubelet_cpu_usage_total",
        "bpf": "kepler_container_bpf_cpu_time_us_total",
    }
    metric_to_validate_power = {
        "rapl": "kepler_node_package_joules_total",
        "platform": "kepler_node_platform_joules_total"
    }
    for metric, query in metric_to_validate_pod.items():
        target_df = validate_df[validate_df["query"]==query]
        valid_df = target_df[target_df[">0"] > 0]
        if len(valid_df) == 0:
            # no data
            continue
        availability = len(valid_df)/len(target_df)
        valid_datapoint = valid_df[">0"].sum()
        item = dict()
        item["usage_metric"] = metric
        item["availability"] = availability
        item["valid_datapoint"] = valid_datapoint
        items += [item]
    summary_df = pd.DataFrame(items)
    print(summary_df)
    for metric, query in metric_to_validate_pod.items():
        target_df = validate_df[validate_df["query"]==query]
        no_data_df = target_df[target_df["count"] == 0]
        zero_data_df = target_df[target_df[">0"] == 0]
        valid_df = target_df[target_df[">0"] > 0]
        print("==== {} ====".format(metric))
        if len(no_data_df) > 0:
            print("{} pods: \tNo data for {}".format(len(no_data_df), pd.unique(no_data_df["scenarioID"])))
        if len(zero_data_df) > 0:
            print("{} pods: \tZero data for {}".format(len(zero_data_df), pd.unique(zero_data_df["scenarioID"])))

        print("{} pods: \tValid\n".format(len(valid_df)))
        print("Valid data points:")
        print( "Empty" if len(valid_df[">0"]) == 0 else valid_df.groupby(["scenarioID"]).sum()[[">0"]])
    for metric, query in metric_to_validate_power.items():
        target_df = validate_df[validate_df["query"]==query]
        print("{} data: \t{}".format(metric, target_df[">0"].values))

def get_validate_df(data_path, benchmark_filename, query_response):
    items = []
    query_results = prom_responses_to_results(query_response)
    container_queries = [query for query in query_results.keys() if "container" in query]
    status_data = load_json(data_path, benchmark_filename)
    if status_data is None or status_data.get("status", None) == None:
        # select all with keyword
        for query in container_queries:
            df = query_results[query]
            if len(df) == 0:
                # set validate item // no value
                item = dict()
                item["pod"] = benchmark_filename
                item["scenarioID"] = ""
                item["query"] = query
                item["count"] = 0
                item[">0"] = 0
                item["total"] = 0
                items += [item]
                continue
            filtered_df = df.copy()
            if "pod_name" in df.columns:
                filtered_df = filtered_df[filtered_df["pod_name"].str.contains(benchmark_filename)]
            # set validate item
            item = dict()
            item["pod"] = benchmark_filename
            item["scenarioID"] = ""
            item["query"] = query
            item["count"] = len(filtered_df)
            item[">0"] = len(filtered_df[filtered_df[query] > 0])
            item["total"] = filtered_df[query].max()
            items += [item]
    else:
        cpe_results = status_data["status"]["results"]
        for result in cpe_results:
            scenarioID = result["scenarioID"]
            scenarios = result["scenarios"]
            configurations = result["configurations"]
            for k, v in scenarios.items():
                result[k] = v
            for k, v in configurations.items():
                result[k] = v
            repetitions = result["repetitions"]
            for rep in repetitions:
                podname = rep["pod"]
                for query in container_queries:
                    df = query_results[query]
                    if len(df) == 0:
                        # set validate item // no value
                        item = dict()
                        item["pod"] = podname
                        item["scenarioID"] = scenarioID
                        item["query"] = query
                        item["count"] = 0
                        item[">0"] = 0
                        item["total"] = 0
                        items += [item]
                        continue
                    filtered_df = df[df["pod_name"]==podname]
                    # set validate item
                    item = dict()
                    item["pod"] = podname
                    item["scenarioID"] = scenarioID
                    item["query"] = query
                    item["count"] = len(filtered_df)
                    item[">0"] = len(filtered_df[filtered_df[query] > 0])
                    item["total"] = filtered_df[query].max()
                    items += [item]
    energy_queries = [query for query in query_results.keys() if "_joules" in query]
    for query in energy_queries:
        df = query_results[query]
        if len(df) == 0:
            # set validate item // no value
            item = dict()
            item["pod"] = ""
            item["scenarioID"] = ""
            item["query"] = query
            item["count"] = 0
            item[">0"] = 0
            item["total"] = 0
            items += [item]
            continue
        # set validate item
        item = dict()
        item["pod"] = ""
        item["scenarioID"] = ""
        item["query"] = query
        item["count"] = len(df)
        item[">0"] = len(df[df[query] > 0])
        item["total"] = df[query].max()
        items += [item]
    other_queries = [query for query in query_results.keys() if query not in container_queries and query not in energy_queries]
    for query in other_queries:
        df = query_results[query]
        if len(df) == 0:
            # set validate item // no value
            item = dict()
            item["pod"] = benchmark_filename
            item["scenarioID"] = ""
            item["query"] = query
            item["count"] = 0
            item[">0"] = 0
            item["total"] = 0
            items += [item]
            continue
        # set validate item
        item = dict()
        item["pod"] = benchmark_filename
        item["scenarioID"] = ""
        item["query"] = query
        item["count"] = len(df)
        item[">0"] = len(df[df[query] > 0])
        item["total"] = df[query].max()
        items += [item]
    validate_df = pd.DataFrame(items)
    if not validate_df.empty:
        print(validate_df.groupby(["scenarioID", "query"]).sum()[["count", ">0"]])
    return validate_df

def check_ot_fg(args, valid_fg):
    ot = None
    fg = None
    if args.output_type:
        try:
            ot = ModelOutputType[args.output_type]
        except KeyError:
            print("invalid output type. please use AbsPower or DynPower", args.output_type)
            exit()
    if args.feature_group:
        valid_fg_name_list = [fg.name for fg in valid_fg]
        try:
            fg = FeatureGroup[args.feature_group]
            if args.feature_group not in valid_fg_name_list:
                print("feature group: {} is not available in your data. please choose from the following list: {}".format(args.feature_group, valid_fg_name_list))
                exit()
        except KeyError:
            print("invalid feature group: {}. valid feature group are {}.".format((args.feature_group, [fg.name for fg in valid_fg])))
            exit()
    return ot, fg

def assert_train(trainer, data, energy_components):
    import pandas as pd
    node_types = pd.unique(data[node_info_column])
    for node_type in node_types:
        node_type_str = int(node_type)
        node_type_filtered_data = data[data[node_info_column] == node_type]
        X_values = node_type_filtered_data[trainer.features].values
        for component in energy_components:
            output = trainer.predict(node_type_str, component, X_values)
            if output is not None:
                assert len(output) == len(X_values), "length of predicted values != features ({}!={})".format(len(output), len(X_values))

def get_isolator(data_path, isolator, profile, pipeline_name, target_hints, bg_hints, abs_pipeline_name):
    pipeline_path = get_pipeline_path(data_path, pipeline_name=pipeline_name)
    from train import MinIdleIsolator, NoneIsolator, DefaultProfiler, ProfileBackgroundIsolator, TrainIsolator, generate_profiles
    supported_isolator = {
        MinIdleIsolator().get_name(): MinIdleIsolator(),
        NoneIsolator().get_name(): NoneIsolator(),
    }

    if target_hints:
        target_hints = target_hints.split(",")
    else:
        target_hints = []

    if bg_hints:
        bg_hints = bg_hints.split(",")
    else:
        bg_hints = []

    profiles = dict()
    if profile:
        idle_response = load_json(data_path, profile)
        idle_data = prom_responses_to_results(idle_response)
        if idle_data is None:
            print("failed to read idle data")
            return None
        profile_map = DefaultProfiler.process(idle_data, profile_top_path=pipeline_path)
        profiles = generate_profiles(profile_map)
        profile_isolator =  ProfileBackgroundIsolator(profiles, idle_data)
        supported_isolator[profile_isolator.get_name()] = profile_isolator
        if abs_pipeline_name != "":
            trainer_isolator = TrainIsolator(idle_data=idle_data, profiler=DefaultProfiler, target_hints=target_hints, bg_hints=bg_hints, abs_pipeline_name=abs_pipeline_name)
            supported_isolator[trainer_isolator.get_name()] = trainer_isolator
    else:
        if abs_pipeline_name != "":
            trainer_isolator = TrainIsolator(target_hints=target_hints, bg_hints=bg_hints, abs_pipeline_name=abs_pipeline_name)
            supported_isolator[trainer_isolator.get_name()] = trainer_isolator

    if isolator not in supported_isolator:
        print("isolator {} is not supported. supported isolator: {}".format(isolator, supported_isolator.keys()))
        return None
    return supported_isolator[isolator]

def get_extractor(extractor):
    from train import DefaultExtractor, SmoothExtractor
    supported_extractor = {
        DefaultExtractor().get_name(): DefaultExtractor(),
        SmoothExtractor().get_name(): SmoothExtractor()
    }
    if extractor not in supported_extractor:
        print("extractor {} is not supported. supported extractor: {}".format(extractor, supported_extractor.keys()))
        return None
    return supported_extractor[extractor]

def get_pipeline(data_path, pipeline_name, extractor, profile, target_hints, bg_hints, abs_pipeline_name, isolator, abs_trainer_names, dyn_trainer_names, energy_sources, valid_feature_groups):
    from train import NewPipeline
    isolator = get_isolator(data_path, isolator, profile, pipeline_name, target_hints, bg_hints, abs_pipeline_name)
    extractor = get_extractor(extractor)
    pipeline = NewPipeline(pipeline_name, abs_trainer_names, dyn_trainer_names, extractor=extractor, isolator=isolator, target_energy_sources=energy_sources ,valid_feature_groups=valid_feature_groups)
    return pipeline