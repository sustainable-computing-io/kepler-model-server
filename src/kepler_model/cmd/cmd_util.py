import datetime
import os

import pandas as pd

from kepler_model.util.loader import default_node_type, get_pipeline_path, load_json
from kepler_model.util.prom_types import (
    SOURCE_COL,
    energy_component_to_query,
    node_info_column,
    prom_responses_to_results,
)
from kepler_model.util.saver import assure_path, save_csv
from kepler_model.util.train_types import FeatureGroup, ModelOutputType, PowerSourceMap

UTC_OFFSET_TIMEDELTA = datetime.datetime.utcnow() - datetime.datetime.now()


def print_file_to_stdout(data_path, args):
    file_path = os.path.join(data_path, args.output)
    try:
        with open(file_path) as file:
            contents = file.read()
            print(contents)
    except FileNotFoundError:
        print(f"Error: Output '{file_path}' not found.")
    except OSError:
        print(f"Error: Unable to read output '{file_path}'.")


def extract_time(data_path, benchmark_filename):
    data = load_json(data_path, benchmark_filename)
    if "metadata" in data:
        start_str = data["metadata"]["creationTimestamp"]
        start = datetime.datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%SZ")
        end_str = data["status"]["results"][-1]["repetitions"][-1]["pushedTime"].split(".")[0]
        end = datetime.datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
    else:
        start_str = data["startTimeUTC"]
        start = datetime.datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%SZ")
        end_str = data["endTimeUTC"]
        end = datetime.datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%SZ")
    print(UTC_OFFSET_TIMEDELTA)
    return start - UTC_OFFSET_TIMEDELTA, end - UTC_OFFSET_TIMEDELTA


def save_query_results(data_path, output_filename, query_response):
    query_results = prom_responses_to_results(query_response)
    save_path = os.path.join(data_path, f"{output_filename}_csv")
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
        # CPU instruction is mainly used for ratio.
        # reference:  https://github.com/sustainable-computing-io/kepler/blob/0b328cf7c79db9a11426fb80a1a922383e40197c/pkg/config/config.go#L92
        "hwc": "kepler_container_cpu_instructions_total",
        "bpf": "kepler_container_bpf_cpu_time_ms_total",
    }
    metric_to_validate_power = {"rapl-sysfs": "kepler_node_package_joules_total", "acpi": "kepler_node_platform_joules_total"}
    for metric, query in metric_to_validate_pod.items():
        target_df = validate_df[validate_df["query"] == query]
        valid_df = target_df[target_df[">0"] > 0]
        if len(valid_df) == 0:
            # no data
            continue
        availability = len(valid_df) / len(target_df)
        valid_datapoint = valid_df[">0"].sum()
        item = dict()
        item["usage_metric"] = metric
        item["availability"] = availability
        item["valid_datapoint"] = valid_datapoint
        items += [item]
    summary_df = pd.DataFrame(items)
    print(summary_df)
    for metric, query in metric_to_validate_pod.items():
        target_df = validate_df[validate_df["query"] == query]
        no_data_df = target_df[target_df["count"] == 0]
        zero_data_df = target_df[target_df[">0"] == 0]
        valid_df = target_df[target_df[">0"] > 0]
        print(f"==== {metric} ====")
        if len(no_data_df) > 0:
            print("{} pods: \tNo data for {}".format(len(no_data_df), pd.unique(no_data_df["scenarioID"])))
        if len(zero_data_df) > 0:
            print("{} pods: \tZero data for {}".format(len(zero_data_df), pd.unique(zero_data_df["scenarioID"])))

        print(f"{len(valid_df)} pods: \tValid\n")
        print("Valid data points:")
        print("Empty" if len(valid_df[">0"]) == 0 else valid_df.groupby(["scenarioID"]).sum()[[">0"]])
    for metric, query in metric_to_validate_power.items():
        target_df = validate_df[validate_df["query"] == query]
        print("{} data: \t{}".format(metric, target_df[">0"].values))


def get_validate_df(data_path, benchmark_filename, query_response):
    items = []
    query_results = prom_responses_to_results(query_response)
    container_queries = [query for query in query_results.keys() if "container" in query]
    print("Container Queries: ", container_queries)
    status_data = load_json(data_path, benchmark_filename)
    filter_by_benchmark = False
    if status_data is None or "status" not in status_data:
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
                # check if we can use inputted benchmark to filtered stressing pods
                podname_filtered = filtered_df[filtered_df["pod_name"].str.contains(benchmark_filename)]
                if len(podname_filtered) > 0:
                    filter_by_benchmark = True
                    filtered_df = podname_filtered
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
        filtered_by_benchmark = True
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
                    filtered_df = df[df["pod_name"] == podname]
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
    print("Energy Queries: ", container_queries)
    for energy_source, energy_components in PowerSourceMap.items():
        for component in energy_components:
            query = energy_component_to_query(component)
            df = query_results[query]
            df = df[df[SOURCE_COL] == energy_source]
            if len(df) == 0:
                # set validate item // no value
                item = dict()
                item["pod"] = ""
                item["scenarioID"] = energy_source
                item["query"] = query
                item["count"] = 0
                item[">0"] = 0
                item["total"] = 0
                items += [item]
                continue
            # set validate item
            item = dict()
            item["pod"] = ""
            item["scenarioID"] = energy_source
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

    if filter_by_benchmark:
        print("===========================================\n Use benchmark name to filter pod results: \n\n", benchmark_filename)
    else:
        print("============================================\n Present results from all pods: \n\n")
    if not validate_df.empty:
        print(validate_df.groupby(["scenarioID", "query"]).sum()[["count", ">0"]])
    else:
        print("Validate dataframe is empty.")
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
                print(f"feature group: {args.feature_group} is not available in your data. please choose from the following list: {valid_fg_name_list}")
                exit()
        except KeyError:
            print("invalid feature group: {}. valid feature group are {}.".format((args.feature_group, [fg.name for fg in valid_fg])))
            exit()
    return ot, fg


import sklearn


def assert_train(trainer, data, energy_components):
    import pandas as pd

    node_types = pd.unique(data[node_info_column])
    for node_type in node_types:
        node_type_filtered_data = data[data[node_info_column] == node_type]
        X_values = node_type_filtered_data[trainer.features].values
        for component in energy_components:
            try:
                output = trainer.predict(node_type, component, X_values)
                if output is not None:
                    assert len(output) == len(X_values), f"length of predicted values != features ({len(output)}!={len(X_values)})"
            except sklearn.exceptions.NotFittedError:
                pass


def get_isolator(data_path, isolator, profile, pipeline_name, target_hints, bg_hints, abs_pipeline_name, replace_node_type=default_node_type):
    pipeline_path = get_pipeline_path(data_path, pipeline_name=pipeline_name)
    from kepler_model.train import (
        DefaultProfiler,
        MinIdleIsolator,
        NoneIsolator,
        ProfileBackgroundIsolator,
        TrainIsolator,
        generate_profiles,
    )

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
        profile_map = DefaultProfiler.process(idle_data, profile_top_path=pipeline_path, replace_node_type=replace_node_type)
        profiles = generate_profiles(profile_map)
        profile_isolator = ProfileBackgroundIsolator(profiles, idle_data)
        supported_isolator[profile_isolator.get_name()] = profile_isolator
        if abs_pipeline_name != "":
            trainer_isolator = TrainIsolator(idle_data=idle_data, profiler=DefaultProfiler, target_hints=target_hints, bg_hints=bg_hints, abs_pipeline_name=abs_pipeline_name)
            supported_isolator[trainer_isolator.get_name()] = trainer_isolator
    elif abs_pipeline_name != "":
        trainer_isolator = TrainIsolator(target_hints=target_hints, bg_hints=bg_hints, abs_pipeline_name=abs_pipeline_name)
        supported_isolator[trainer_isolator.get_name()] = trainer_isolator

    if isolator not in supported_isolator:
        print(f"isolator {isolator} is not supported. supported isolator: {supported_isolator.keys()}")
        return None
    return supported_isolator[isolator]


def get_extractor(extractor):
    from kepler_model.train import DefaultExtractor, SmoothExtractor

    supported_extractor = {DefaultExtractor().get_name(): DefaultExtractor(), SmoothExtractor().get_name(): SmoothExtractor()}
    if extractor not in supported_extractor:
        print(f"extractor {extractor} is not supported. supported extractor: {supported_extractor.keys()}")
        return None
    return supported_extractor[extractor]


def get_pipeline(data_path, pipeline_name, extractor, profile, target_hints, bg_hints, abs_pipeline_name, isolator, abs_trainer_names, dyn_trainer_names, energy_sources, valid_feature_groups, replace_node_type=default_node_type):
    from kepler_model.train import NewPipeline

    isolator = get_isolator(data_path, isolator, profile, pipeline_name, target_hints, bg_hints, abs_pipeline_name, replace_node_type=replace_node_type)
    extractor = get_extractor(extractor)
    pipeline = NewPipeline(pipeline_name, abs_trainer_names, dyn_trainer_names, extractor=extractor, isolator=isolator, target_energy_sources=energy_sources, valid_feature_groups=valid_feature_groups)
    return pipeline
