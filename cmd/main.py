import os
import sys
import argparse
import datetime
import pandas as pd

data_path = "/data"
default_output_filename = "output"
default_trainer_names = [ 'PolynomialRegressionTrainer', 'GradientBoostingRegressorTrainer', 'SGDRegressorTrainer', 'KNeighborsRegressorTrainer', 'LinearRegressionTrainer','SVRRegressorTrainer']
default_trainers = ",".join(default_trainer_names)

UTC_OFFSET_TIMEDELTA = datetime.datetime.utcnow() - datetime.datetime.now()
data_path = os.getenv("CPE_DATAPATH", data_path)

# set model top path to data path
os.environ['MODEL_PATH'] = data_path

cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)

from util.prom_types import PROM_SERVER, PROM_QUERY_INTERVAL, PROM_QUERY_STEP, PROM_HEADERS, PROM_SSL_DISABLE
from util.prom_types import metric_prefix as KEPLER_METRIC_PREFIX, node_info_column, prom_responses_to_results, TIMESTAMP_COL
from util.train_types import ModelOutputType, FeatureGroup
from util.loader import load_json, DEFAULT_PIPELINE, load_pipeline_metadata, get_pipeline_path, get_model_group_path, list_pipelines, list_model_names, load_metadata
from util.saver import save_json, save_csv
from util.config import ERROR_KEY
from util import get_valid_feature_group_from_queries, PowerSourceMap
from train.prom.prom_query import _range_queries

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

def get_pipeline(pipeline_name, extractor, profile, isolator, abs_trainer_names, dyn_trainer_names, energy_sources, valid_feature_groups):
    pipeline_path = get_pipeline_path(data_path, pipeline_name=pipeline_name)
    from train import DefaultExtractor, SmoothExtractor, MinIdleIsolator, NoneIsolator, DefaultProfiler, ProfileBackgroundIsolator, TrainIsolator, generate_profiles, NewPipeline
    supported_extractor = {
        "default": DefaultExtractor(),
        "smooth": SmoothExtractor()
    }
    supported_isolator = {
        "min": MinIdleIsolator(),
        "none": NoneIsolator(),
        MinIdleIsolator.__name__: MinIdleIsolator(),
        NoneIsolator.__name__: NoneIsolator()
    }

    profiles = dict()
    if profile:
        idle_response = load_json(data_path, profile)
        idle_data = prom_responses_to_results(idle_response)
        if idle_data is None:
            print("failed to read idle data")
            return None
        profile_map = DefaultProfiler.process(idle_data, profile_top_path=pipeline_path)
        profiles = generate_profiles(profile_map)
        supported_isolator["profile"] = ProfileBackgroundIsolator(profiles, idle_data)
        supported_isolator[ProfileBackgroundIsolator.__name__] = supported_isolator["profile"]
        supported_isolator["trainer"] = TrainIsolator(idle_data=idle_data, profiler=DefaultProfiler)
        supported_isolator[TrainIsolator.__name__] = supported_isolator["trainer"]

    if isolator not in supported_isolator:
        print("isolator {} is not supported. supported isolator: {}".format(isolator, supported_isolator.keys()))
        return None
    
    if extractor not in supported_extractor:
        print("extractor {} is not supported. supported extractor: {}".format(isolator, supported_extractor.keys()))
        return None

    isolator = supported_isolator[isolator]
    extractor = supported_extractor[extractor]
    pipeline = NewPipeline(pipeline_name, abs_trainer_names, dyn_trainer_names, extractor=extractor, isolator=isolator, target_energy_sources=energy_sources ,valid_feature_groups=valid_feature_groups)
    return pipeline

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
    pipeline = get_pipeline(pipeline_name, args.extractor, args.profile, args.isolator, abs_trainer_names, dyn_trainer_names, energy_sources, valid_feature_groups)
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
                    if trainer.node_level:
                        assert_train(trainer, abs_data, energy_components)
                    else:
                        assert_train(trainer, dyn_data, energy_components)
            # save data
            data_saved_path = os.path.join(pipeline.path, "preprocessed_data")
            save_csv(data_saved_path, "{}_{}_abs_data".format(energy_source, feature_group.name), abs_data)
            save_csv(data_saved_path, "{}_{}_dyn_data".format(energy_source, feature_group.name), dyn_data)


        print("=========== Train {} Summary ============".format(energy_source))
        # save args 
        argparse_dict = vars(args)
        save_json(pipeline.path, "train_arguments", argparse_dict)
        print("Train args:", argparse_dict)
        # save metadata
        pipeline.save_metadata()
        # save pipeline
        pipeline.archive_pipeline()
        print_cols = ["feature_group", "model_name", "mae"]
        print("AbsPower pipeline results:")
        metadata_df = load_pipeline_metadata(pipeline.path, energy_source, ModelOutputType.AbsPower.name)
        print(metadata_df.sort_values(by=ERROR_KEY)[print_cols])
        print("DynPower pipeline results:")
        metadata_df = load_pipeline_metadata(pipeline.path, energy_source, ModelOutputType.DynPower.name)
        print(metadata_df.sort_values(by=ERROR_KEY)[print_cols])

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
    valid_fg_name_list = [fg.name for fg in valid_fg]
    if args.feature_group:
        try:
            fg = FeatureGroup[args.feature_group]
            if args.feature_group not in valid_fg_name_list:
                print("feature group: {} is not available in your data. please choose from the following list: {}".format(args.feature_group, valid_fg_name_list))
                exit()
            valid_fg = [fg]
        except KeyError:
            print("invalid feature group: {}. valid feature group are {}.", (args.feature_group, [fg.name for fg in valid_fg]))
            exit()
    try:
        ot = ModelOutputType[args.output_type]
    except KeyError:
        print("invalid output type. please use AbsPower or DynPower", args.output_type)
        exit()

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
            pipeline = get_pipeline(pipeline_name, args.extractor, args.profile, pipeline_metadata["isolator"], pipeline_metadata["abs_trainers"], pipeline_metadata["dyn_trainers"], energy_sources, valid_fg)
            if pipeline is None:
                print("cannot get pipeline {}.".format(pipeline_name))
                continue
            for fg in  valid_fg:
                print(" Feature Group: ", fg)
                abs_data, dyn_data, power_labels = pipeline.prepare_data_from_input_list(input_query_results_list, energy_components, energy_source, fg.name)
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
                    if args.output_type == ModelOutputType.AbsPower:
                        data = abs_data
                    else:
                        data = dyn_data
                    predicted_power_map, data_with_prediction = model.append_prediction(data)
                    max_mae = None
                    for energy_component, _ in predicted_power_map.items():
                        label_power_columns = [col for col in power_labels if energy_component in col]
                        predicted_power_colname = default_predicted_col_func(energy_component)
                        sum_power_label = data.groupby([TIMESTAMP_COL]).mean()[label_power_columns].sum(axis=1).sort_index()
                        sum_predicted_power = data_with_prediction.groupby([TIMESTAMP_COL]).sum().sort_index()[predicted_power_colname]
                        mae, mse = compute_error(sum_power_label, sum_predicted_power)
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
            shutil.make_archive(os.path.join(output_folder, best_model), 'zip', best_model_path)
            # save result
            estimation_result = "{}_estimation_result".format(energy_source)
            save_csv(output_folder, estimation_result, best_result)
    
if __name__ == "__main__":
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
    parser.add_argument("--isolator", type=str, help="Specify isolator name (none, min, profile, trainer).", default="none")
    parser.add_argument("--profile", type=str, help="Specify profile input (required for trainer and profile isolator).")
    parser.add_argument("-e", "--energy-source", type=str, help="Specify energy source.", default="rapl")
    parser.add_argument("--abs-trainers", type=str, help="Specify trainer names (use comma(,) as delimiter).", default=default_trainers)
    parser.add_argument("--dyn-trainers", type=str, help="Specify trainer names (use comma(,) as delimiter).", default=default_trainers)

    # Validate arguments
    parser.add_argument("--benchmark", type=str, help="Specify benchmark file name.")

    # Estimate arguments
    parser.add_argument("-ot", "--output-type", type=str, help="Specify output type (AbsPower or DynPower) for energy estimation.", default="AbsPower")
    parser.add_argument("-fg", "--feature-group", type=str, help="Specify target feature group for energy estimation.", default="")
    parser.add_argument("--model-name", type=str, help="Specify target model name for energy estimation.")
    

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

    