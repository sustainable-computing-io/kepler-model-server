import os
import sys
import argparse
import datetime
import pandas as pd

data_path = "/data"
default_output_filename = "output"

data_path = os.getenv("DATAPATH", data_path)

cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)

from util.prom_types import PROM_SERVER, PROM_QUERY_INTERVAL, PROM_QUERY_STEP, PROM_QUERY_START_TIME, PROM_QUERY_END_TIME, PROM_HEADERS, PROM_SSL_DISABLE, PROM_THIRDPARTY_METRICS
from util.prom_types import metric_prefix as KEPLER_METRIC_PREFIX, prom_responses_to_results, TIMESTAMP_COL, feature_to_query, update_thirdparty_metrics, node_info_column
from util.extract_types import get_expected_power_columns
from util.train_types import ModelOutputType, FeatureGroups, is_single_source_feature_group, all_feature_groups, default_trainers
from util.loader import load_json, DEFAULT_PIPELINE, load_pipeline_metadata, get_pipeline_path, get_model_group_path, list_pipelines, list_model_names, load_metadata, load_csv, get_preprocess_folder, get_general_filename, load_machine_spec
from util.saver import save_json, save_csv, save_train_args, _pipeline_model_metadata_filename
from util.config import ERROR_KEY, model_toppath
from util import get_valid_feature_group_from_queries, PowerSourceMap
from train.prom.prom_query import _range_queries
from train.exporter import exporter
from train import load_class
from train.profiler.node_type_index import NodeTypeIndexCollection, NodeTypeSpec, generate_spec

from cmd_plot import ts_plot, feature_power_plot, summary_plot, metadata_plot
from cmd_util import extract_time, save_query_results, get_validate_df, summary_validation, get_extractor, check_ot_fg, get_pipeline, assert_train, get_isolator, UTC_OFFSET_TIMEDELTA

import threading

"""
query

    make a query to prometheus metric server for a specified set of queries (with metric prefix) and save as a map of metric name to dataframe containing labels and value of the metric.


arguments:
- --server : specify prometheus server URL (PROM_HEADERS and PROM_SSL_DISABLE configuration might be set via configmap or environment variables if needed)
- --output : specify query output file name.  There will be two files generated: [output].json and [output]_validate_result.csv for kepler query response and validated results.
- --metric-prefix : specify prefix to filter target query metrics (default: kepler)
- --thirdparty-metrics : specify list of third party metrics to query in addition to the metrics with specified metric prefix (optional)
- Time range of query can be specified by either of the following choices
    - --input : file containing data of start time and end time which could be either of the following format
                - CPE benchmark resource in json if you run workload with CPE-operator (https://github.com/IBM/cpe-operator)
                - custom benchmark in json with `startTimeUTC` and `endTimeUTC` data
    - --benchmark : file to save query timestamp according to either of the following raw parameters
        - --start-time, --end-time : start time and end time (date +%Y-%m-%dT%H:%M:%SZ) 
        - --interval : last interval in second
    * The priority is input > start-time,end-time > interval
- --to-csv : to save converted query result in csv format
- --id : specify machine ID
"""

def query(args):
    if not args.id:
        args.id = "unknown"
        print("Machine ID has not defined by --id, use `unknown`")
    machine_id = args.id
    generate_spec(data_path, machine_id)
    from prometheus_api_client import PrometheusConnect
    prom = PrometheusConnect(url=args.server, headers=PROM_HEADERS, disable_ssl=PROM_SSL_DISABLE)
    start = None
    end = None
    if args.input:
        benchmark_filename = args.input
        filepath = os.path.join(data_path, benchmark_filename+".json")
        if os.path.isfile(filepath):
            print("Query from {}.".format(benchmark_filename))
            start, end = extract_time(data_path, benchmark_filename)
    if start is None or end is None:
        if args.benchmark:
            benchmark_filename = args.benchmark
        else:
            print("Please provide either input (for providing timestamp file) or benchmark (for saving query timestamp)")
            exit()
        if args.start_time != "" and args.end_time != "":
            # by [start time, end time]
            print("Query from start_time {} to end_time {}.".format(args.start_time, args.end_time))
            start = datetime.datetime.strptime(args.start_time, '%Y-%m-%dT%H:%M:%SZ')
            end = datetime.datetime.strptime(args.end_time , '%Y-%m-%dT%H:%M:%SZ')
        else:
            # by interval
            print("Query last {} interval.".format(args.interval))
            end = datetime.datetime.now(datetime.timezone.utc)
            start = end - datetime.timedelta(seconds=args.interval)
        # save benchmark
        item = dict()
        item["startTimeUTC"] = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        item["endTimeUTC"] = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        save_json(path=data_path, name=benchmark_filename, data=item)
        start = datetime.datetime.strptime(item["startTimeUTC"], '%Y-%m-%dT%H:%M:%SZ') - UTC_OFFSET_TIMEDELTA
        end = datetime.datetime.strptime(item["endTimeUTC"], '%Y-%m-%dT%H:%M:%SZ') - UTC_OFFSET_TIMEDELTA

    available_metrics = prom.all_metrics()

    queries = None
    if args.thirdparty_metrics != "":
        queries = [m for m in available_metrics if args.metric_prefix in m or m in args.thirdparty_metrics]
    elif PROM_THIRDPARTY_METRICS != [""]:
        queries = [m for m in available_metrics if args.metric_prefix in m or m in PROM_THIRDPARTY_METRICS]
    else:
        queries = [m for m in available_metrics if args.metric_prefix in m]

    print("Start {} End {}".format(start, end))
    response = _range_queries(prom, queries, start, end, args.step, None)
    save_json(path=data_path, name=args.output, data=response)
    if args.to_csv:
        save_query_results(data_path, args.output, response)
    # try validation if applicable
    validate_df = get_validate_df(data_path, benchmark_filename, response)
    summary_validation(validate_df)
    save_csv(path=data_path, name=args.output + "_validate_result", data=validate_df)

"""
validate

    quickly validate each queried metric that how many data it contains and how many of them have non-zero values based on benchmark file

arguments:
- --input : specify kepler query response file (output of `query` function)
- --output : specify output file name. The output will be in csv format
- --benchmark : if benchmark is based on CPE, the validated result will be groupped by the scenario. 
                Otherwise, the validated result will be an accumulated of all containers.
"""

def validate(args):
    response_filename = args.input
    response = load_json(data_path, response_filename)
    validate_df = get_validate_df(data_path, args.benchmark, response)
    summary_validation(validate_df)
    if args.output:
        save_csv(path=data_path, name=args.output, data=validate_df)

"""
extract

    parse kepler query response, filter only target feature group and energy source, and calculate per-second (guage) value at each timestamp 

arguments:
- --input : specify kepler query response file (output of `query` function) as an input to the extractor
- --output : specify extracted file name. There will be two files generated: extracted_[output].csv and extracted_[output]_raw.csv for guage and counter data, respectively.
- --extractor : specify extractor class (default or smooth)
- --feature-group : specify target feature group (check https://sustainable-computing.io/kepler_model_server/pipeline/#feature-group)
- --energy-source : specify target energy source (check https://sustainable-computing.io/kepler_model_server/pipeline/#energy-source)
- --output-type : specify target output type, (default = AbsPower)
                  - AbsPower (index = timestamp), DynPower (index = timestamp and container ID)
                  - check https://sustainable-computing.io/kepler_model_server/pipeline/#power-isolation
- --thirdparty-metrics : specify list of third party metric to export (required only for ThirdParty feature group)
"""

def extract(args):
    extractor = get_extractor(args.extractor)
    # single input
    input = args.input
    response = load_json(data_path, input)
    query_results = prom_responses_to_results(response)
    # Inject thirdparty_metrics to FeatureGroup
    if args.thirdparty_metrics != "":
        update_thirdparty_metrics(args.thirdparty_metrics)
    elif PROM_THIRDPARTY_METRICS != [""]:
        update_thirdparty_metrics(PROM_THIRDPARTY_METRICS)
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
        print("extract {} train data to {}".format(args.output_type, "extracted_" + args.output))
    return feature_power_data, power_cols

"""
isolate

    extract data from kepler query and remove idle/background power from the power columns (check https://sustainable-computing.io/kepler_model_server/pipeline/#power-isolation)

arguments:
- --input : specify kepler query response file (output of `query` function)
- --output : specify extracted file name. 
    There will be three files generated:
     - extracted_[output].csv for extracted data (gauge values)
     - extracted_[output]_raw.csv for raw extracted data (counter values)
     - isolated_[output].csv for isolated data
- --extractor : specify extractor class (default or smooth)
- --isolator : specify isolator class (none, profile, min, trainer)
- --feature-group : specify target feature group (check https://sustainable-computing.io/kepler_model_server/pipeline/#feature-group)
- --energy-source : specify target energy source (check https://sustainable-computing.io/kepler_model_server/pipeline/#energy-source)
- --output-type : specify target extracting output type, (default = AbsPower)
                  - AbsPower (index = timestamp), DynPower (index = timestamp and container ID)
                  - check https://sustainable-computing.io/kepler_model_server/pipeline/#power-isolation
- --thirdparty-metrics : specify list of third party metric to export (required only for ThirdParty feature group)
- --abs-pipeline-name : specify pipeline name to be used for initializing trainer isolator (only required for trainer isolator)
- --profile : specify kepler query result when running no workload (required only for profile isolator)
- --pipeline-name : specify output pipeline name to save processed profile data (only required when --profile is set)
- Hints required only for trainer isolator to separate background process can be specified by either of 
    - --target-hints : specify target process keywords (pod's substring with comma delimiter) to keep in DynPower model training
    - --bg-hints : specify background process keywords to remove from DynPower model training
    * If both are defined, target-hints will be considered first.
"""

def isolate(args):
    extracted_data, power_labels = extract(args)
    if extracted_data is None or power_labels is None:
        return None
    pipeline_name = DEFAULT_PIPELINE if not args.pipeline_name else args.pipeline_name
    isolator = get_isolator(data_path, args.isolator, args.profile, pipeline_name, args.target_hints, args.bg_hints, args.abs_pipeline_name)
    isolated_data = isolator.isolate(extracted_data, label_cols=power_labels, energy_source=args.energy_source)
    if args.output:
        save_csv(data_path, "isolated_" + args.output, isolated_data)
        print("isolate train data to {}".format("isolated_" + args.output))

"""
isolate_from_data

    remove idle/background power from the power columns read from extracted data from `extract` function output

arguments:
- --input : specify extracted output file
- --output : specify isolated file name. The output filename is isolated_[output].csv
- --isolator : specify isolator class (none, profile, min, trainer)
- --feature-group : specify target feature group (check https://sustainable-computing.io/kepler_model_server/pipeline/#feature-group)
- --energy-source : specify target energy source (check https://sustainable-computing.io/kepler_model_server/pipeline/#energy-source)
- --thirdparty-metrics : specify list of third party metric to export (required only for ThirdParty feature group)
- --abs-pipeline-name : specify pipeline name to be used for initializing trainer isolator (only required for trainer isolator)
- --profile : specify kepler query result when running no workload (required only for profile isolator)
- --pipeline-name : specify output pipeline name to save processed profile data (only required when --profile is set)
- Hints required only for trainer isolator to separate background process can be specified by either of 
    - --target-hints : specify target process keywords (pod's substring with comma delimiter) to keep in DynPower model training
    - --bg-hints : specify background process keywords to remove from DynPower model training
    * If both are defined, target-hints will be considered first.
"""

def isolate_from_data(args):
    energy_components = PowerSourceMap[args.energy_source]
    extracted_data = load_csv(data_path, "extracted_" + args.input)
    power_columns = get_expected_power_columns(energy_components=energy_components)
    pipeline_name = DEFAULT_PIPELINE if not args.pipeline_name else args.pipeline_name
    isolator = get_isolator(data_path, args.isolator, args.profile, pipeline_name, args.target_hints, args.bg_hints, args.abs_pipeline_name)
    isolated_data = isolator.isolate(extracted_data, label_cols=power_columns, energy_source=args.energy_source)
    if args.output:
        save_csv(data_path, "isolated_" + args.output, isolated_data)

"""
train_from_data

    train extracted/isolated data for AbsPower/DynPower models from `extract` or `isolate`/`isolate_from_data` function output

arguments:
- --input : specify extracted/isolated output file
- --pipeline-name : specify output pipeline name. 
        - The trained model will be saved in folder [pipeline_name]/[energy_source]/[output_type]/[feature_group]/[model_name]_[node_type]
- --feature-group : specify target feature group (check https://sustainable-computing.io/kepler_model_server/pipeline/#feature-group)
- --energy-source : specify target energy source (check https://sustainable-computing.io/kepler_model_server/pipeline/#energy-source)
- --output-type : specify target output type (check https://sustainable-computing.io/kepler_model_server/pipeline/#power-isolation) - default: AbsPower
- --trainers : specify trainer names (use comma(,) as delimiter, default: XgboostFitTrainer)
- --thirdparty-metrics : specify list of third party metric to export (required only for ThirdParty feature group)
"""

def train_from_data(args):

    # Inject thirdparty_metrics to FeatureGroup
    if args.thirdparty_metrics != "":
        update_thirdparty_metrics(args.thirdparty_metrics)
    elif PROM_THIRDPARTY_METRICS != [""]:
        update_thirdparty_metrics(PROM_THIRDPARTY_METRICS)
    valid_fg = [fg_key for fg_key in FeatureGroups.keys()]
    ot, fg = check_ot_fg(args, valid_fg)
    if fg is None or ot is None:
        print("feature group {} or model output type {} is wrong. (valid feature group: {})".format(args.feature_group, args.output_type, all_feature_groups))
        exit()

    energy_components = PowerSourceMap[args.energy_source]
    node_level=False
    if ot == ModelOutputType.AbsPower:
        node_level=True

    data = load_csv(data_path, args.input)
    power_columns = get_expected_power_columns(energy_components=energy_components)

    energy_components = PowerSourceMap[args.energy_source]

    node_type = None
    node_collection = None
    if args.id:
        machine_id = args.id
        pipeline_path = get_pipeline_path(model_toppath, pipeline_name=args.pipeline_name)
        node_collection = NodeTypeIndexCollection(pipeline_path)
        machine_spec_json = load_machine_spec(data_path, machine_id)
        if machine_spec_json is not None:
            new_spec = NodeTypeSpec()
            new_spec.load(machine_spec_json)
            node_type = node_collection.index_train_machine(machine_id, new_spec)
            print("Replace {} with {}".format(node_info_column, node_type))
            data[node_info_column] = int(node_type)

    if node_type is None:
        print("Machine ID has not defined by --id or machine spec is not available, do not auto-replace node_type")

    trainers =  args.trainers.split(",")
    metadata_list = []
    for trainer in trainers:
        trainer_class = load_class("trainer", trainer)
        trainer = trainer_class(energy_components, args.feature_group, args.energy_source, node_level=node_level, pipeline_name=args.pipeline_name)
        trainer.process(data, power_columns, pipeline_lock=threading.Lock())
        assert_train(trainer, data, energy_components)
        metadata = trainer.get_metadata()
        metadata_list += [metadata]
    metadata_df = pd.concat(metadata_list)
    print_cols = ["model_name", "mae", "mape"]
    print(metadata_df[print_cols])

    if node_collection is not None:
        print("Save node index")
        node_collection.save()
    
"""
train

    automate a pipeline process to kepler query input from `query` function output to all available feature groups for both AbsPower and DynPower

arguments:
- --input : specify kepler query response file (output of `query` function) 
- --pipeline-name : specify output pipeline name. 
        - The trained model will be saved in folder [pipeline_name]/[energy_source]/[output_type]/[feature_group]/[model_name]_[node_type]
- --extractor : specify extractor class (default or smooth)
- --isolator : specify isolator class (none, profile, min, trainer)
- --abs-pipeline-name : specify pipeline name to be used for initializing trainer isolator (only required for trainer isolator)
- --profile : specify kepler query result when running no workload (required only for profile isolator)
- Hints required only for trainer isolator to separate background process can be specified by either of 
    - --target-hints : specify target process keywords (pod's substring with comma delimiter) to keep in DynPower model training
    - --bg-hints : specify background process keywords to remove from DynPower model training
    * If both are defined, target-hints will be considered first.
- --abs-trainers : specify a list of trainers for training AbsPower models (use comma(,) as delimiter) - default: apply all available trainers
- --dyn-trainers : specify a list of trainers for training DynPower models (use comma(,) as delimiter) - default: apply all available trainers
- --energy-source : specify target energy sources (use comma(,) as delimiter) 
- --thirdparty-metrics : specify list of third party metric to export (required only for ThirdParty feature group)
- --id : specify machine ID 
"""

def train(args):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    if not args.input:
        print("must give input filename (query response) via --input for training.")
        exit()

    # Inject thirdparty_metrics to FeatureGroup
    if args.thirdparty_metrics != "":
        update_thirdparty_metrics(args.thirdparty_metrics)
    elif PROM_THIRDPARTY_METRICS != [""]:
        update_thirdparty_metrics(PROM_THIRDPARTY_METRICS)

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

    if args.dyn_trainers == "default":
        args.dyn_trainers = default_trainers
    if args.abs_trainers == "default":
        args.abs_trainers = default_trainers

    abs_trainer_names = args.abs_trainers.split(",")
    dyn_trainer_names = args.dyn_trainers.split(",")
    
    node_type=None
    if args.id:
        machine_id = args.id
        pipeline = get_pipeline(data_path, pipeline_name, args.extractor, args.profile, args.target_hints, args.bg_hints, args.abs_pipeline_name, args.isolator, abs_trainer_names, dyn_trainer_names, energy_sources, valid_feature_groups)
        machine_spec_json = load_machine_spec(data_path, machine_id)
        if machine_spec_json is not None:
            new_spec = NodeTypeSpec()
            new_spec.load(machine_spec_json)
            node_type = pipeline.node_collection.index_train_machine(machine_id, new_spec)
            node_type = int(node_type)

    if node_type is None:
        print("Machine ID has not defined by --id or machine spec is not available, do not auto-replace node_type")
    
    if pipeline is None:
        print("cannot get pipeline")
        exit()
    for energy_source in energy_sources:
        energy_components = PowerSourceMap[energy_source]
        for feature_group in valid_feature_groups:
            success, abs_data, dyn_data = pipeline.process_multiple_query(input_query_results_list, energy_components, energy_source, feature_group=feature_group.name, replace_node_type=node_type)
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
        # save node collection
        pipeline.node_collection.save()
        # save pipeline
        pipeline.archive_pipeline()
        print_cols = ["feature_group", "model_name", "mae", "mape"]
        print("AbsPower pipeline results:")
        metadata_df = load_pipeline_metadata(pipeline.path, energy_source, ModelOutputType.AbsPower.name)
        if metadata_df is not None:
            print(metadata_df.sort_values(by=[ERROR_KEY])[print_cols])
        print("DynPower pipeline results:")
        metadata_df = load_pipeline_metadata(pipeline.path, energy_source, ModelOutputType.DynPower.name)
        if metadata_df is not None:
            print(metadata_df.sort_values(by=[ERROR_KEY])[print_cols])

    warnings.resetwarnings()

"""
estimate

    apply trained model of specified pipeline to predict power consumption from kepler metrics
    
arguments:
- --input : specify kepler query response file (output of `query` function) 
- --pipeline-name : specify trained pipeline name. 
- --model-name : specified model name (optional, applying all models if not specified)
- --energy-source : specify target energy sources (use comma(,) as delimiter) 
- --output-type : specify target output type (check https://sustainable-computing.io/kepler_model_server/pipeline/#power-isolation) - default: AbsPower
- --thirdparty-metrics : specify list of third party metric to export (required only for ThirdParty feature group)
- --profile : specify kepler query result when running no workload (required only for pipeline with profile isolator)
- Hints required only for pipeline with trainer isolator to separate background process can be specified by either of 
    - --target-hints : specify target process keywords (pod's substring with comma delimiter) to keep in DynPower model training
    - --bg-hints : specify background process keywords to remove from DynPower model training
    * If both are defined, target-hints will be considered first.
"""

def estimate(args):
    if not args.input:
        print("must give input filename (query response) via --input for estimation.")
        exit()

    from estimate import load_model, default_predicted_col_func, compute_error

    # Inject thirdparty_metrics to FeatureGroup
    if args.thirdparty_metrics != "":
        update_thirdparty_metrics(args.thirdparty_metrics)
    elif PROM_THIRDPARTY_METRICS != [""]:
        update_thirdparty_metrics(PROM_THIRDPARTY_METRICS)

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
            pipeline = get_pipeline(data_path, pipeline_name, pipeline_metadata["extractor"], args.profile, args.target_hints, args.bg_hints, args.abs_pipeline_name, pipeline_metadata["isolator"], pipeline_metadata["abs_trainers"], pipeline_metadata["dyn_trainers"], energy_sources, valid_fg)
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
                        mae, mse, mape = compute_error(sum_power_label, sum_predicted_power)
                        summary_item = dict()
                        summary_item["MAE"] = mae
                        summary_item["MSE"] = mse
                        summary_item["MAPE"] = mape
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

"""
plot

    plot data (required results from `extract` and `isolate` functions)
    
arguments:
- --target_data : specify target data (preprocess, estimate, error)
    - `preprocess` plots time series of usage and power metrics for both AbsPower and DynPower
    - `estimate` passes all arguments to `estimate` function, and plots the predicted time series and correlation between usage and power metrics
    - `error` passes all arguments to `estimate` function, and plots the summary of prediction error
    - `metadata` plot pipeline metadata 
- --energy-source : specify target energy sources (use comma(,) as delimiter) 
- --extractor : specify extractor to get preprocessed data of AbsPower model linked to the input data
- --isolator : specify isolator to get preprocessed data of DynPower model linked to the input data
- --pipeline_name : specify pipeline name
"""

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
        os.makedirs(output_folder, exist_ok=True)
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
                ts_plot(feature_data, feature_cols, "Feature group: {}".format(fg.name), output_folder, data_filename)
                if not energy_plot:
                    power_data = data.groupby([TIMESTAMP_COL]).max()
                    data_filename = get_general_filename(args.target_data, energy_source, None, ot, args.extractor, args.isolator)
                    ts_plot(power_data, power_cols, "Power source: {}".format(energy_source), output_folder, data_filename, ylabel="Power (W)")
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
            ts_plot(data, cols, "{} {} Prediction Result \n by {}".format(energy_source, ot.name, model_id), output_folder, "{}_{}".format(data_filename, model_id), subtitles=subtitles, labels=plot_labels, ylabel="Power (W)")
            # plot correlation to utilization if feature group is set
            if fg is not None:
                feature_cols = FeatureGroups[fg]
                scaler = MaxAbsScaler()
                data[feature_cols] = best_restult[[TIMESTAMP_COL] + feature_cols].groupby([TIMESTAMP_COL]).sum().sort_index()
                data[feature_cols] = scaler.fit_transform(data[feature_cols])
                feature_power_plot(data, model_id, ot.name, energy_source, feature_cols, actual_power_cols, predicted_power_cols, output_folder, "{}_{}_corr".format(data_filename, model_id))
    elif args.target_data == "error":
        from estimate import default_predicted_col_func
        from sklearn.preprocessing import MaxAbsScaler
        _, _, _, summary_df = estimate(args)
        for energy_source in energy_sources:
            data_filename = get_general_filename(args.target_data, energy_source, fg, ot, args.extractor, args.isolator)
            summary_plot(args, energy_source, summary_df, output_folder, data_filename)
    elif args.target_data == "metadata":
        for energy_source in energy_sources:
            data_filename = _pipeline_model_metadata_filename(energy_source, ot.name)
            pipeline_path = get_pipeline_path(data_path, pipeline_name=pipeline_name)
            model_metadata_df = load_pipeline_metadata(pipeline_path, energy_source, ot.name)
            metadata_plot(args, energy_source, model_metadata_df, output_folder, data_filename)

"""
export

    export preprocessed data and trained models to the kepler-model-db path
    
arguments:
- --pipeline-name : specify pipeline name that contains the trained models
- --output : specify kepler-model-db/models in local path
- --publisher : specify publisher (github) account
- --benchmark : specify benchmark file that contains data of start time and end time which could be either of the following format
                - CPE benchmark resource in json if you run workload with CPE-operator (https://github.com/IBM/cpe-operator)
                - custom benchmark in json with `startTimeUTC` and `endTimeUTC` data
- --collect-date : specify collection time manually in UTC
- --input : specify kepler query response file (output of `query` function) - optional
"""

def export(args):

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

    if args.benchmark:
        collect_date, _ = extract_time(data_path, args.benchmark)
    elif args.collect_date:
        collect_date = args.collect_date
    else:
        print("need to specify --benchmark or --collect-date")
        exit()

    inputs = []
    if args.input:
        inputs = args.input.split(",")

    pipeline_name = args.pipeline_name
    pipeline_path = get_pipeline_path(data_path, pipeline_name=pipeline_name)

    local_export_path = exporter.export(data_path, pipeline_path, output_path, publisher=args.publisher, collect_date=collect_date, inputs=inputs)
    args.target_data = "metadata"

    args.output = local_export_path
    args.output_type = "AbsPower"
    args.energy_source = ",".join(PowerSourceMap.keys())
    plot(args)
    args.output_type = "DynPower"
    plot(args)

"""
plot_scenario

    separatedly plot data on specified scenario. This function is now limited to only CPE-based benchmark.

arguments:
- --benchmark : specify CPE benchmark resource in json file
- Please refer to `plot` function for the rest arguments.
"""

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
            ts_plot(feature_data, feature_cols, "Feature group: {} ({})".format(fg.name, args.scenario), output_folder, data_filename)
            if not energy_plot:
                power_data = data.groupby([TIMESTAMP_COL]).max()
                data_filename = get_general_filename(args.target_data, energy_source, None, ot, args.extractor, args.isolator) + "_" + args.scenario
                ts_plot(power_data, power_cols, "Power source: {} ({})".format(energy_source, args.scenario), output_folder, data_filename, ylabel="Power (W)")

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Kepler model server entrypoint")
    parser.add_argument("command", type=str, help="The command to execute.")

    parser.add_argument("--data-path", type=str, help="Specify datapath.", default=data_path)

    # Common arguments
    parser.add_argument("-i", "--input", type=str, help="Specify input file/folder name.", default="")
    parser.add_argument("-o", "--output", type=str, help="Specify output file/folder name", default=default_output_filename)

    # Query arguments
    parser.add_argument("-s", "--server", type=str, help="Specify prometheus server.", default=PROM_SERVER)
    parser.add_argument("--interval", type=int, help="Specify query interval.", default=PROM_QUERY_INTERVAL)
    parser.add_argument("--start-time", type=str, help="Specify query start time.", default=PROM_QUERY_START_TIME)
    parser.add_argument("--end-time", type=str, help="Specify query end time.", default=PROM_QUERY_END_TIME)
    parser.add_argument("--step", type=str, help="Specify query step.", default=PROM_QUERY_STEP)
    parser.add_argument("--metric-prefix", type=str, help="Specify metrix prefix to filter.", default=KEPLER_METRIC_PREFIX)
    parser.add_argument("-tm", "--thirdparty-metrics", nargs='+', help="Specify the thirdparty metrics that are not included by Kepler", default="")
    parser.add_argument("--to-csv", type=bool, help="To save converted query response to csv format", default=False)

    # Train arguments
    parser.add_argument("-p", "--pipeline-name", type=str, help="Specify pipeline name.")
    parser.add_argument("--extractor", type=str, help="Specify extractor name (default, smooth).", default="default")
    parser.add_argument("--isolator", type=str, help="Specify isolator name (none, min, profile, trainer).", default="min")
    parser.add_argument("--profile", type=str, help="Specify profile input (required for trainer and profile isolator).")
    parser.add_argument("--target-hints", type=str, help="Specify dynamic workload container name hints (used by TrainIsolator)")
    parser.add_argument("--bg-hints", type=str, help="Specify background workload container name hints (used by TrainIsolator)")
    parser.add_argument("--abs-pipeline-name", type=str, help="Specify AbsPower model pipeline (used by TrainIsolator)", default="")
    parser.add_argument("-e", "--energy-source", type=str, help="Specify energy source.", default="intel_rapl")
    parser.add_argument("--abs-trainers", type=str, help="Specify trainer names for train command (use comma(,) as delimiter).", default="default")
    parser.add_argument("--dyn-trainers", type=str, help="Specify trainer names for train command (use comma(,) as delimiter).", default="default")
    parser.add_argument("--trainers", type=str, help="Specify trainer names for train_from_data command (use comma(,) as delimiter).", default="XgboostFitTrainer")

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
    parser.add_argument("--publisher", type=str, help="Specify github account of model publisher")
    parser.add_argument("--include-raw", type=bool, help="Include raw query data")
    parser.add_argument("--collect-date", type=str, help="Specify collect date directly")

    parser.add_argument("--id", type=str, help="specify machine id")

    # Parse the command-line arguments
    args = parser.parse_args()

    # set model top path to data path
    data_path = args.data_path

    # Check if the required argument is provided
    if not args.command:
        parser.print_help()
    else:
        if not os.path.exists(data_path):
            if args.command == "query":
                os.makedirs(data_path)
                print("create new folder for data: {}".format(data_path))
            else:
                print("{0} not exists. For docker run, {0} must be mount, add -v \"$(pwd)\":{0}. For native run, set DATAPATH".format(data_path))
                exit()
        getattr(sys.modules[__name__], args.command)(args)


