import os
import sys
import argparse
import datetime
import pandas as pd

data_path = "/data"
default_output_filename = "output"
default_trainer_names = [ 'PolynomialRegressionTrainer', 'GradientBoostingRegressorTrainer', 'SGDRegressorTrainer', 'KNeighborsRegressorTrainer', 'LinearRegressionTrainer','SVRRegressorTrainer']
default_trainers = ",".join(default_trainer_names)
default_version = "v0.6"

UTC_OFFSET_TIMEDELTA = datetime.datetime.utcnow() - datetime.datetime.now()
data_path = os.getenv("CPE_DATAPATH", data_path)

cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)

from util.prom_types import PROM_SERVER, PROM_QUERY_INTERVAL, PROM_QUERY_STEP, PROM_HEADERS, PROM_SSL_DISABLE
from util.prom_types import metric_prefix as KEPLER_METRIC_PREFIX, node_info_column, prom_responses_to_results, TIMESTAMP_COL, feature_to_query
from util.train_types import ModelOutputType, FeatureGroup, FeatureGroups, is_single_source_feature_group, SingleSourceFeatures
from util.loader import load_json, DEFAULT_PIPELINE, load_pipeline_metadata, get_pipeline_path, get_model_group_path, list_pipelines, list_model_names, load_metadata, load_csv, get_machine_path, get_preprocess_folder, get_general_filename
from util.saver import save_json, save_csv, save_train_args
from util.config import ERROR_KEY
from util import get_valid_feature_group_from_queries, PowerSourceMap
from train.prom.prom_query import _range_queries
from train.exporter import exporter


def print_file_to_stdout(args):
    file_path = os.path.join(data_path, args.output)
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
            print(contents)
    except FileNotFoundError:
        print(f"Error: Output '{file_path}' not found.")
    except IOError:
        print(f"Error: Unable to read output '{file_path}'.")

def extract_time(benchmark_filename):
    data = load_json(data_path, benchmark_filename)
    start_str = data["metadata"]["creationTimestamp"]
    start = datetime.datetime.strptime(start_str, '%Y-%m-%dT%H:%M:%SZ')
    end_str = data["status"]["results"][-1]["repetitions"][-1]["pushedTime"].split(".")[0]
    end = datetime.datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
    print(UTC_OFFSET_TIMEDELTA)
    return start-UTC_OFFSET_TIMEDELTA, end-UTC_OFFSET_TIMEDELTA

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

def get_validate_df(benchmark_filename, query_response):
    items = []
    query_results = prom_responses_to_results(query_response)
    queries = [query for query in query_results.keys() if "container" in query]
    status_data = load_json(data_path, benchmark_filename)
    if status_data is None:
        # select all with keyword
        for query in queries:
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
            filtered_df = df[df["pod_name"].str.contains(benchmark_filename)]
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
                for query in queries:
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
        filtered_df = df.copy()
        # set validate item
        item = dict()
        item["pod"] = ""
        item["scenarioID"] = ""
        item["query"] = query
        item["count"] = len(filtered_df)
        item[">0"] = len(filtered_df[filtered_df[query] > 0])
        item["total"] = filtered_df[query].max()
        items += [item]
    validate_df = pd.DataFrame(items)
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

def query(args):
    from prometheus_api_client import PrometheusConnect
    prom = PrometheusConnect(url=args.server, headers=PROM_HEADERS, disable_ssl=PROM_SSL_DISABLE)
    benchmark_filename = args.input
    if benchmark_filename == "":
        print("Query last {} interval.".format(args.interval))
        end = datetime.datetime.now()
        start = end - datetime.timedelta(seconds=args.interval)
    else:
        print("Query from {}.".format(benchmark_filename))
        start, end = extract_time(benchmark_filename)
    available_metrics = prom.all_metrics()
    queries = [m for m in available_metrics if args.metric_prefix in m]
    print("Start {} End {}".format(start, end))
    response = _range_queries(prom, queries, start, end, args.step, None)
    save_json(path=data_path, name=args.output, data=response)
    # try validation if applicable
    if benchmark_filename != "" and args.metric_prefix == KEPLER_METRIC_PREFIX:
        validate_df = get_validate_df(benchmark_filename, response)
        summary_validation(validate_df)
        save_csv(path=data_path, name=args.output + "_validate_result", data=validate_df)

def validate(args):
    if not args.benchmark:
        print("Need --benchmark")
        exit()

    response_filename = args.input
    response = load_json(data_path, response_filename)
    validate_df = get_validate_df(args.benchmark, response)
    summary_validation(validate_df)
    if args.output:
        save_csv(path=data_path, name=args.output, data=validate_df)
    
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

def get_isolator(isolator, profile, pipeline_name, target_hints, bg_hints):
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
        trainer_isolator = TrainIsolator(idle_data=idle_data, profiler=DefaultProfiler, target_hints=target_hints, bg_hints=bg_hints, abs_pipeline_name=pipeline_name)
        supported_isolator[profile_isolator.get_name()] = profile_isolator
    else:
        trainer_isolator = TrainIsolator(target_hints=target_hints, bg_hints=bg_hints, abs_pipeline_name=pipeline_name)
        
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

def get_pipeline(pipeline_name, extractor, profile, target_hints, bg_hints, isolator, abs_trainer_names, dyn_trainer_names, energy_sources, valid_feature_groups):
    from train import NewPipeline
    isolator = get_isolator(isolator, profile, pipeline_name, target_hints, bg_hints)
    extractor = get_extractor(extractor)
    pipeline = NewPipeline(pipeline_name, abs_trainer_names, dyn_trainer_names, extractor=extractor, isolator=isolator, target_energy_sources=energy_sources ,valid_feature_groups=valid_feature_groups)
    return pipeline

def extract(args):
    extractor = get_extractor(args.extractor)
    # single input
    input = args.input
    response = load_json(data_path, input)
    query_results = prom_responses_to_results(response)

    valid_fg = get_valid_feature_group_from_queries([query for query in query_results.keys() if len(query_results[query]) > 1 ])
    ot, fg = check_ot_fg(args, valid_fg)
    if fg is None or ot is None:
        print("feature group {} or model output type {} is wrong. (valid feature group: {})".format(args.feature_group, args.output_type, valid_fg))
        exit()

    energy_components = PowerSourceMap[args.energy_source]
    node_level=False
    if ot == ModelOutputType.AbsPower:
        node_level=True
    feature_power_data, power_cols, _, _ = extractor.extract(query_results, energy_components, args.feature_group, args.energy_source, node_level=node_level)
    if args.output:
        save_csv(data_path, "extracted_" + args.output, feature_power_data)
        query = feature_to_query(FeatureGroups[fg][0])
        raw_data = query_results[query][[TIMESTAMP_COL, query]].groupby([TIMESTAMP_COL]).sum()
        save_csv(data_path, "extracted_" + args.output[0:-4]+"_raw.csv", raw_data)
    return feature_power_data, power_cols

def isolate(args):
    extracted_data, power_labels = extract(args)
    if extracted_data is None or power_labels is None:
        return None
    pipeline_name = DEFAULT_PIPELINE if not args.pipeline_name else args.pipeline_name
    isolator = get_isolator(args.isolator, args.profile, pipeline_name, args.target_hints, args.bg_hints)
    isolated_data = isolator.isolate(extracted_data, label_cols=power_labels, energy_source=args.energy_source)
    if args.output:
        save_csv(data_path, "isolated_" + args.output, isolated_data)

def train(args):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    if not args.input:
        print("must give input filename (query response) via --input for training.")
        exit()

    pipeline_name = DEFAULT_PIPELINE
    if args.pipeline_name:
       pipeline_name = args.pipeline_name 

    inputs = args.input.split(",")
    energy_sources = args.energy_source.split(",")
    input_query_results_list = []
    valid_feature_groups = None
    for input in inputs:
        response = load_json(data_path, input)
        query_results = prom_responses_to_results(response)

        valid_fg = get_valid_feature_group_from_queries([query for query in query_results.keys() if len(query_results[query]) > 1 ])
        print("valid feature group: ", valid_fg)
        if valid_feature_groups is None:
            valid_feature_groups = valid_fg
        else:
            valid_feature_groups = list(set(valid_feature_groups).intersection(set(valid_fg)))
        input_query_results_list += [query_results]

    
    abs_trainer_names = args.abs_trainers.split(",")
    dyn_trainer_names = args.dyn_trainers.split(",")
    pipeline = get_pipeline(pipeline_name, args.extractor, args.profile, args.target_hints, args.bg_hints, args.isolator, abs_trainer_names, dyn_trainer_names, energy_sources, valid_feature_groups)
    if pipeline is None:
        print("cannot get pipeline")
        exit()
    for energy_source in energy_sources:
        energy_components = PowerSourceMap[energy_source]
        for feature_group in valid_feature_groups:
            success, abs_data, dyn_data = pipeline.process_multiple_query(input_query_results_list, energy_components, energy_source, feature_group=feature_group.name)
            assert success, "failed to process pipeline {}".format(pipeline.name) 
            for trainer in pipeline.trainers:
                if trainer.feature_group == feature_group and trainer.energy_source == energy_source:
                    if trainer.node_level and abs_data is not None:
                        assert_train(trainer, abs_data, energy_components)
                    elif dyn_data is not None:
                        assert_train(trainer, dyn_data, energy_components)
            # save data
            data_saved_path = get_preprocess_folder(pipeline.path)
            if abs_data is not None:
                save_csv(data_saved_path, get_general_filename("preprocess", energy_source, feature_group, ModelOutputType.AbsPower, args.extractor), abs_data)
            if dyn_data is not None:
                save_csv(data_saved_path, get_general_filename("preprocess", energy_source, feature_group, ModelOutputType.DynPower, args.extractor, args.isolator), dyn_data)


        print("=========== Train {} Summary ============".format(energy_source))
        # save args 
        argparse_dict = vars(args)
        save_train_args(pipeline.path, argparse_dict)
        print("Train args:", argparse_dict)
        # save metadata
        pipeline.save_metadata()
        # save pipeline
        pipeline.archive_pipeline()
        print_cols = ["feature_group", "model_name", "mae"]
        print("AbsPower pipeline results:")
        metadata_df = load_pipeline_metadata(pipeline.path, energy_source, ModelOutputType.AbsPower.name)
        if metadata_df is not None:
            print(metadata_df.sort_values(by=[ERROR_KEY])[print_cols])
        print("DynPower pipeline results:")
        metadata_df = load_pipeline_metadata(pipeline.path, energy_source, ModelOutputType.DynPower.name)
        if metadata_df is not None:
            print(metadata_df.sort_values(by=[ERROR_KEY])[print_cols])

        warnings.resetwarnings()

def estimate(args):
    if not args.input:
        print("must give input filename (query response) via --input for estimation.")
        exit()

    from estimate import load_model, default_predicted_col_func, compute_error

    inputs = args.input.split(",")
    energy_sources = args.energy_source.split(",")
    input_query_results_list = []
    for input in inputs:
        response = load_json(data_path, input)
        query_results = prom_responses_to_results(response)
        input_query_results_list += [query_results]
    
    valid_fg = get_valid_feature_group_from_queries([query for query in query_results.keys() if len(query_results[query]) > 1 ])
    ot, fg = check_ot_fg(args, valid_fg)
    if fg is not None: 
        valid_fg = [fg]

    best_result_map = dict()
    power_labels_map = dict()
    best_model_id_map = dict()
    summary_items = []
    for energy_source in energy_sources:
        if args.pipeline_name:
            pipeline_names = [args.pipeline_name]
        else:
            pipeline_names = list_pipelines(data_path, energy_source, args.output_type)
        energy_components = PowerSourceMap[energy_source]
        best_result = None
        best_model_path = None
        best_mae = None
        print("Pipelines: ", pipeline_names)
        for pipeline_name in pipeline_names:
            pipeline_path = get_pipeline_path(data_path, pipeline_name=pipeline_name)
            pipeline_metadata = load_metadata(pipeline_path)
            if pipeline_metadata is None:
                print("no metadata for pipeline {}.".format(pipeline_name))
                continue
            pipeline = get_pipeline(pipeline_name, args.extractor, args.profile, args.target_hints, args.bg_hints, pipeline_metadata["isolator"], pipeline_metadata["abs_trainers"], pipeline_metadata["dyn_trainers"], energy_sources, valid_fg)
            if pipeline is None:
                print("cannot get pipeline {}.".format(pipeline_name))
                continue
            for fg in  valid_fg:
                print(" Feature Group: ", fg)
                abs_data, dyn_data, power_labels = pipeline.prepare_data_from_input_list(input_query_results_list, energy_components, energy_source, fg.name)
                if energy_source not in power_labels_map:
                    power_labels_map[energy_source] = power_labels
                group_path = get_model_group_path(data_path, ot, fg, energy_source, assure=False, pipeline_name=pipeline_name)
                model_names = list_model_names(group_path)
                if args.model_name:
                    if args.model_name not in model_names:
                        print("model: {} is not availble in pipeline {}, continue. available models are {}".format(args.model_name, pipeline_name, model_names))
                        continue
                    model_names = [args.model_name]
                for model_name in model_names:
                    model_path = os.path.join(group_path, model_name)
                    model = load_model(model_path)
                    if args.output_type == ModelOutputType.AbsPower.name:
                        data = abs_data
                    else:
                        data = dyn_data
                    predicted_power_map, data_with_prediction = model.append_prediction(data)
                    max_mae = None
                    for energy_component, _ in predicted_power_map.items():
                        predicted_power_colname = default_predicted_col_func(energy_component)
                        label_power_columns = [col for col in power_labels if energy_component in col and col != predicted_power_colname]
                        sum_power_label = data.groupby([TIMESTAMP_COL]).mean()[label_power_columns].sum(axis=1).sort_index()
                        sum_predicted_power = data_with_prediction.groupby([TIMESTAMP_COL]).sum().sort_index()[predicted_power_colname]
                        mae, mse = compute_error(sum_power_label, sum_predicted_power)
                        summary_item = dict()
                        summary_item["MAE"] = mae
                        summary_item["MSE"] = mse
                        summary_item["n"] = len(sum_predicted_power)
                        summary_item["energy_component"] = energy_component
                        summary_item["energy_source"] = energy_source
                        summary_item["Model"] = model_name
                        summary_item["Feature Group"] = fg.name
                        summary_items += [summary_item]
                        if max_mae is None or mae > max_mae:
                            max_mae = mae
                    if best_mae is None or max_mae < best_mae:
                        # update best
                        best_result = data_with_prediction.copy()
                        best_model_path = model_path
                        best_mae = max_mae
                    print("     Model {}: ".format(model_name), max_mae)

        # save best result
        if best_model_path is not None:
            print("Energy consumption of energy source {} is predicted by {}".format(energy_source, best_model_path.replace(data_path, "")))
            print("MAE = ", best_mae)
            output_folder = os.path.join(data_path, args.output)
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            # save model
            import shutil
            best_model = "{}_model".format(energy_source)
            if not args.id:
                # not only for export
                shutil.make_archive(os.path.join(output_folder, best_model), 'zip', best_model_path)
                # save result
                estimation_result = "{}_estimation_result".format(energy_source)
                save_csv(output_folder, estimation_result, best_result)
            best_result_map[energy_source] = best_result
            path_splits = best_model_path.split("/")
            best_model_id_map[energy_source] = "{} using {}".format(path_splits[-1], path_splits[-2])
    return best_result_map, power_labels_map, best_model_id_map, pd.DataFrame(summary_items)
    
def _ts_plot(data, cols, title, output_folder, name, labels=None, subtitles=None, ylabel=None):
    plot_height = 3
    plot_width = 10
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(font_scale=1.2)
    fig, axes = plt.subplots(len(cols), 1, figsize=(plot_width, len(cols)*plot_height))
    for i in range(0, len(cols)):
        if len(cols) == 1:
            ax = axes
        else:
            ax = axes[i]
        if isinstance(cols[i], list):
            # multiple lines
            for j in range(0, len(cols[i])):
                sns.lineplot(data=data, x=TIMESTAMP_COL, y=cols[i][j], ax=ax, label=labels[j])
            ax.set_title(subtitles[i])
        else:
            sns.lineplot(data=data, x=TIMESTAMP_COL, y=cols[i], ax=ax)
            ax.set_title(cols[i])
        if ylabel is not None:
            ax.set_ylabel(ylabel)
    plt.suptitle(title, x=0.5, y=0.99)
    plt.tight_layout()
    filename = os.path.join(output_folder, name + ".png")
    fig.savefig(filename)
    plt.close()

def _feature_power_plot(data, model_id, output_type, energy_source, feature_cols, actual_power_cols, predicted_power_cols, output_folder, name):
    plot_height = 5
    plot_width = 5
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(font_scale=1.2)
    row_num = len(feature_cols)
    col_num = len(actual_power_cols)
    width = max(10, col_num*plot_width)
    fig, axes = plt.subplots(row_num, col_num, figsize=(width, row_num*plot_height))
    for xi in range(0, row_num):
        feature_col = feature_cols[xi]
        for yi in range(0, col_num):
            if row_num == 1:
                if col_num == 1:
                    ax = axes
                else:
                    ax = axes[yi]
            else:
                if col_num == 1:
                    ax = axes[xi]
                else:
                    ax = axes[xi][yi]
            sorted_data = data.sort_values(by=[feature_col])
            sns.scatterplot(data=sorted_data, x=feature_col, y=actual_power_cols[yi], ax=ax, label="actual")
            sns.lineplot(data=sorted_data, x=feature_col, y=predicted_power_cols[yi], ax=ax, label="predicted", color='C1')
            if xi == 0:
                ax.set_title(actual_power_cols[yi])
            if yi == 0:
                ax.set_ylabel("Power (W)")
    title = "{} {} prediction correlation \n by {}".format(energy_source, output_type, model_id)
    plt.suptitle(title, x=0.5, y=0.99)
    plt.tight_layout()
    filename = os.path.join(output_folder, name + ".png")
    fig.savefig(filename)
    plt.close()

def _summary_plot(energy_source, summary_df, output_folder, name):
    if len(summary_df) == 0:
        print("no summary data to plot")
        return
    
    plot_height = 3
    plot_width = 20
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(font_scale=1.2)

    energy_components = PowerSourceMap[energy_source]
    col_num = len(energy_components)
    fig, axes = plt.subplots(col_num, 1, figsize=(plot_width, plot_height*col_num))
    for i in range(0, col_num):
        component = energy_components[i]
        data = summary_df[(summary_df["energy_source"]==energy_source) & (summary_df["energy_component"]==component)]
        data = data.sort_values(by=["Feature Group", "MAE"])
        if col_num == 1:
            ax = axes
        else:
            ax = axes[i]
        sns.barplot(data=data, x="Feature Group", y="MAE", hue="Model", ax=ax)
        ax.set_title(component)
        ax.set_ylabel("MAE (Watt)")
        ax.set_ylim((0, 100))
        if i < col_num-1:
            ax.set_xlabel("")
        ax.legend(bbox_to_anchor=(1.05, 1.05))
    plt.suptitle("{} {} error".format(energy_source, args.output_type))
    plt.tight_layout()
    filename = os.path.join(output_folder, name + ".png")
    fig.savefig(filename)
    plt.close()

def plot(args):
    pipeline_name = DEFAULT_PIPELINE if not args.pipeline_name else args.pipeline_name
    pipeline_path = get_pipeline_path(data_path, pipeline_name=pipeline_name)
    if not args.target_data:
        print("must give target data via --target-data to plot.")
        exit()
    valid_fg = [fg_key for fg_key in FeatureGroups.keys()]
    ot, fg = check_ot_fg(args, valid_fg)
    if fg is not None: 
        valid_fg = [fg]
    print("Plot:", args)
    energy_sources = args.energy_source.split(",")
    output_folder = os.path.join(data_path, args.output)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if args.target_data == "preprocess":
        data_saved_path = get_preprocess_folder(pipeline_path)
        feature_plot = []
        for energy_source in energy_sources:
            energy_plot = False
            for fg in valid_fg:
                if (len(valid_fg) > 1 and not is_single_source_feature_group(fg)) or (energy_plot and fg.name in feature_plot):
                    # no need to plot if it is a mixed source or already plotted
                    continue
                data_filename = get_general_filename(args.target_data, energy_source, fg, ot, args.extractor, args.isolator)
                if data_filename is None:
                    print("cannot get preprocessed data for ", ot.name)
                    return
                data = load_csv(data_saved_path, data_filename)
                if data is None:
                    print("cannot load data from {}/{}".format(data_saved_path, data_filename))
                    continue
                feature_plot += [fg.name]
                feature_cols = FeatureGroups[fg] 
                power_cols = [col for col in data.columns if "power" in col]
                feature_data = data.groupby([TIMESTAMP_COL]).sum()
                _ts_plot(feature_data[feature_data[feature_cols]>0], feature_cols, "Feature group: {}".format(fg.name), output_folder, data_filename)
                if not energy_plot:
                    power_data = data.groupby([TIMESTAMP_COL]).max()
                    data_filename = get_general_filename(args.target_data, energy_source, None, ot, args.extractor, args.isolator)
                    _ts_plot(power_data, power_cols, "Power source: {}".format(energy_source), output_folder, data_filename, ylabel="Power (W)")
    elif args.target_data == "estimate":
        from estimate import default_predicted_col_func
        from sklearn.preprocessing import MaxAbsScaler

        best_result_map, power_labels_map, best_model_id_map, summary_df = estimate(args)
        print(summary_df)
        for energy_source, best_restult in best_result_map.items():
            best_restult = best_restult.reset_index()
            power_labels = power_labels_map[energy_source]
            model_id = best_model_id_map[energy_source]
            subtitles = []
            cols = []
            plot_labels = ["actual", "predicted"]
            data = pd.DataFrame()
            actual_power_cols = []
            predicted_power_cols = []
            for energy_component in PowerSourceMap[energy_source]:
                subtitles += [energy_component]
                predicted_power_colname = default_predicted_col_func(energy_component)
                label_power_columns = [col for col in power_labels if energy_component in col and col != predicted_power_colname]
                data[energy_component] = best_restult.groupby([TIMESTAMP_COL]).mean()[label_power_columns].sum(axis=1).sort_index()
                data[predicted_power_colname] = best_restult.groupby([TIMESTAMP_COL]).sum().sort_index()[predicted_power_colname]
                cols += [[energy_component, predicted_power_colname]]
                actual_power_cols += [energy_component]
                predicted_power_cols += [predicted_power_colname]
            data_filename = get_general_filename(args.target_data, energy_source, fg, ot, args.extractor, args.isolator)
            # plot prediction
            _ts_plot(data, cols, "{} {} Prediction Result \n by {}".format(energy_source, ot.name, model_id), output_folder, "{}_{}".format(data_filename, model_id), subtitles=subtitles, labels=plot_labels, ylabel="Power (W)")
            # plot correlation to utilization if feature group is set
            if fg is not None:
                feature_cols = FeatureGroups[fg]
                scaler = MaxAbsScaler()
                data[feature_cols] = best_restult[[TIMESTAMP_COL] + feature_cols].groupby([TIMESTAMP_COL]).sum().sort_index()
                data[feature_cols] = scaler.fit_transform(data[feature_cols])
                _feature_power_plot(data, model_id, ot.name, energy_source, feature_cols, actual_power_cols, predicted_power_cols, output_folder, "{}_{}_corr".format(data_filename, model_id))
    elif args.target_data == "error":
        from estimate import default_predicted_col_func
        from sklearn.preprocessing import MaxAbsScaler
        _, _, _, summary_df = estimate(args)
        for energy_source in energy_sources:
            data_filename = get_general_filename(args.target_data, energy_source, fg, ot, args.extractor, args.isolator)
            _summary_plot(energy_source, summary_df, output_folder, data_filename)

def export(args):
    if not args.id:
        print("need to specify --id")
        exit()

    if not args.pipeline_name:
        print("need to specify pipeline name via -p or --pipeline-name")
        exit()

    if args.output == default_output_filename:
        print("need to specify --output for /models path")
        exit()
    output_path = args.output  

    if not args.publisher:
        print("need to specify --publisher")
        exit()

    if not args.benchmark:
        print("need to specify --benchmark to extract collection time")
        exit()

    pipeline_name = args.pipeline_name 
    machine_id = args.id
    pipeline_path = get_pipeline_path(data_path, pipeline_name=pipeline_name)
    machine_path = get_machine_path(output_path, args.version, machine_id)

    collect_date, _ = extract_time(args.benchmark)
    exporter.export(data_path, pipeline_path, machine_path, machine_id=machine_id, version=args.version, publisher=args.publisher, collect_date=collect_date, include_raw=args.include_raw)

    args.energy_source = ",".join(PowerSourceMap.keys())

    for ot in ModelOutputType:
        args.output_type = ot.name

        # plot preprocess data
        args.target_data = "preprocess"
        args.output = get_preprocess_folder(machine_path)
        plot(args)
        
        # plot error
        args.target_data = "error"
        args.output = os.path.join(machine_path, "error_summary")
        plot(args)


    args.target_data = "estimate"
    args.output = os.path.join(machine_path, "best_estimation")
    for ot in ModelOutputType:
        args.output_type = ot.name
        # plot estimate
        for feature_group in SingleSourceFeatures:
            args.feature_group = feature_group
            plot(args)
            
def plot_scenario(args):
    if not args.benchmark:
        print("Need --benchmark")
        exit()

    if not args.scenario:
        print("Need --scenario")
        exit()

    # filter scenario
    input_scenarios = args.scenario.split(",")
    status_data = load_json(data_path, args.benchmark)
    target_pods = []
    cpe_results = status_data["status"]["results"]
    for result in cpe_results:
        scenarioID = result["scenarioID"]
        target = False
        for scenario in input_scenarios:
            if scenario in scenarioID:
                target = True
                break
        if not target:
            continue

        scenarios = result["scenarios"]
        configurations = result["configurations"]
        for k, v in scenarios.items():
            result[k] = v
        for k, v in configurations.items():
            result[k] = v
        repetitions = result["repetitions"]
        for rep in repetitions:
            podname = rep["pod"]
            target_pods += [podname]

    response = load_json(data_path, args.input)
    query_results = prom_responses_to_results(response)
    for query, data in query_results.items():
        if "pod_name" in data.columns:
            query_results[query]  = data[data["pod_name"].isin(target_pods)]

    valid_fg = [fg_key for fg_key in FeatureGroups.keys()]
    ot, fg = check_ot_fg(args, valid_fg)
    if fg is not None: 
        valid_fg = [fg]
    energy_sources = args.energy_source.split(",")
    output_folder = os.path.join(data_path, args.output)

    print("Plot:", args)
    feature_plot = []
    for energy_source in energy_sources:
        energy_components = PowerSourceMap[energy_source]
        energy_plot = False
        for fg in valid_fg:
            if (len(valid_fg) > 1 and not is_single_source_feature_group(fg)) or (energy_plot and fg.name in feature_plot):
                # no need to plot if it is a mixed source or already plotted
                continue
            data_filename = get_general_filename(args.target_data, energy_source, fg, ot, args.extractor, args.isolator) + "_" + args.scenario
            if data_filename is None:
                print("cannot get preprocessed data for ", ot.name)
                return
            from train import DefaultExtractor

            extractor = DefaultExtractor()
            data, power_cols, _, _ = extractor.extract(query_results, energy_components, fg.name, args.energy_source, node_level=True)
            feature_plot += [fg.name]
            feature_cols = FeatureGroups[fg] 
            power_cols = [col for col in data.columns if "power" in col]
            feature_data = data.groupby([TIMESTAMP_COL]).sum()
            _ts_plot(feature_data, feature_cols, "Feature group: {} ({})".format(fg.name, args.scenario), output_folder, data_filename)
            if not energy_plot:
                power_data = data.groupby([TIMESTAMP_COL]).max()
                data_filename = get_general_filename(args.target_data, energy_source, None, ot, args.extractor, args.isolator) + "_" + args.scenario
                _ts_plot(power_data, power_cols, "Power source: {} ({})".format(energy_source, args.scenario), output_folder, data_filename, ylabel="Power (W)")

if __name__ == "__main__":
    # set model top path to data path
    os.environ['MODEL_PATH'] = data_path

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Kepler model server entrypoint")
    parser.add_argument("command", type=str, help="The command to execute.")

    # Common arguments
    parser.add_argument("-i", "--input", type=str, help="Specify input file/folder name.", default="")
    parser.add_argument("-o", "--output", type=str, help="Specify output file/folder name", default=default_output_filename)

    # Query arguments
    parser.add_argument("-s", "--server", type=str, help="Specify prometheus server.", default=PROM_SERVER)
    parser.add_argument("--interval", type=int, help="Specify query interval.", default=PROM_QUERY_INTERVAL)
    parser.add_argument("--step", type=str, help="Specify query step.", default=PROM_QUERY_STEP)
    parser.add_argument("--metric-prefix", type=str, help="Specify metrix prefix to filter.", default=KEPLER_METRIC_PREFIX)

    # Train arguments
    parser.add_argument("-p", "--pipeline-name", type=str, help="Specify pipeline name.")
    parser.add_argument("--extractor", type=str, help="Specify extractor name (default, smooth).", default="default")
    parser.add_argument("--isolator", type=str, help="Specify isolator name (none, min, profile, trainer).", default="min")
    parser.add_argument("--profile", type=str, help="Specify profile input (required for trainer and profile isolator).")
    parser.add_argument("--target-hints", type=str, help="Specify dynamic workload container name hints (used by TrainIsolator)")
    parser.add_argument("--bg-hints", type=str, help="Specify background workload container name hints (used by TrainIsolator)")
    parser.add_argument("-e", "--energy-source", type=str, help="Specify energy source.", default="rapl")
    parser.add_argument("--abs-trainers", type=str, help="Specify trainer names (use comma(,) as delimiter).", default=default_trainers)
    parser.add_argument("--dyn-trainers", type=str, help="Specify trainer names (use comma(,) as delimiter).", default=default_trainers)

    # Validate arguments
    parser.add_argument("--benchmark", type=str, help="Specify benchmark file name.")

    # Estimate arguments
    parser.add_argument("-ot", "--output-type", type=str, help="Specify output type (AbsPower or DynPower) for energy estimation.", default="AbsPower")
    parser.add_argument("-fg", "--feature-group", type=str, help="Specify target feature group for energy estimation.", default="")
    parser.add_argument("--model-name", type=str, help="Specify target model name for energy estimation.")

    # Plot arguments
    parser.add_argument("--target-data", type=str, help="Speficy target plot data (preprocess, estimate)")
    parser.add_argument("--scenario", type=str, help="Speficy scenario")
    
    # Export arguments
    parser.add_argument("--id", type=str, help="specify machine id")
    parser.add_argument("--version", type=str, help="Specify model server version.", default=default_version)
    parser.add_argument("--publisher", type=str, help="Specify github account of model publisher")
    parser.add_argument("--include-raw", type=bool, help="Include raw query data")

    # Parse the command-line arguments
    args = parser.parse_args()

    if not os.path.exists(data_path):
        print("{} must be mount, add -v \"$(pwd)\":{} .".format(data_path, data_path))
        exit()

    # Check if the required argument is provided
    if not args.command:
        parser.print_help()
    else:
        getattr(sys.modules[__name__], args.command)(args)

    