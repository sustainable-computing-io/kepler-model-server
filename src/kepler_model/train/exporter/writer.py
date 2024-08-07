import os
import pandas as pd


from kepler_model.util.loader import load_json, version
from kepler_model.util.saver import assure_path, _pipeline_model_metadata_filename, _power_curve_filename
from kepler_model.train.exporter.validator import mae_threshold, mape_threshold
from kepler_model.util.train_types import ModelOutputType, PowerSourceMap

error_report_foldername = "error_report"


def write_markdown(markdown_filepath, markdown_content):
    try:
        with open(markdown_filepath, "w", encoding="utf-8") as markdown_file:
            # Write the Markdown content to the file
            markdown_file.write(markdown_content)
            print(f"Markdown file '{markdown_filepath}' has been created successfully.")
    except IOError as e:
        print(f"Cannot write '{markdown_filepath}': {e}")


# Function to convert a dataframe to a Markdown table
def data_to_markdown_table(data):
    # Get the column headers
    headers = list(data.keys())

    # Initialize the Markdown table with headers
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    # Iterate through the dictionary and add rows to the table
    for i in range(len(data[headers[0]])):
        row = "| " + " | ".join([str(data[key][i]) for key in headers]) + " |\n"
        markdown_table += row

    return markdown_table


def format_cpe_content(data):
    spec = data["spec"]
    iterations = spec["iterationSpec"]["iterations"]
    items = dict()
    for iteration in iterations:
        items[iteration["name"]] = iteration["values"]
    df = pd.DataFrame(items)
    content = data_to_markdown_table(df)
    content += "\nrepetition: {}".format(spec["repetition"])
    return content


def get_workload_content(data_path, inputs):
    workload_content = ""

    for input in inputs:
        content = None
        benchmark_name = input
        if "_kepler_query" in input:
            benchmark_name = input.replace("_kepler_query", "")
        data = load_json(data_path, benchmark_name)
        if data is not None:
            content = format_cpe_content(data)
        else:
            # read file directly
            filepath = os.path.join(data_path, input)
            if os.path.exists(filepath):
                with open(filepath, "r") as file:
                    content = file.read()

        workload_content += """
### {}

<-- put workload description here -->

<details>

{}

</details>

        """.format(benchmark_name.split(".")[0], content)
    return workload_content


def format_trainer(trainers):
    trainer_content = ""
    for trainer in trainers:
        trainer_content += "  - {}\n".format(trainer)
    return trainer_content


def generate_pipeline_page(version_path, pipeline_metadata, workload_content, skip_if_exist=True):
    doc_path = os.path.join(version_path, ".doc")
    assure_path(doc_path)
    pipeline_name = pipeline_metadata["name"]
    markdown_filename = "{}.md".format(pipeline_name)
    markdown_filepath = os.path.join(doc_path, markdown_filename)
    if skip_if_exist and os.path.exists(markdown_filepath):
        print(f"Markdown file '{markdown_filepath}' already exists.")
        return

    markdown_content = """
# Pipeline {} 

## Description

<-- put pipeline description here -->

## Components

- **Extractor:** {}
- **Isolator:** {}
- **AbsPower Trainers:**

{}

- **DynPower Trainers:** 

{}

## Workload information

{}
    """.format(pipeline_name, pipeline_metadata["extractor"], pipeline_metadata["isolator"], format_trainer(pipeline_metadata["abs_trainers"]), "  (same as AbsPower Trainers)" if pipeline_metadata["abs_trainers"] == pipeline_metadata["dyn_trainers"] else pipeline_metadata["dyn_trainers"], workload_content)

    write_markdown(markdown_filepath, markdown_content)


# error_report_url called by generate_pipeline_readme
def _error_report_url(export_path, node_type, assure):
    error_report_folder = os.path.join(export_path, error_report_foldername)
    if assure:
        assure_path(error_report_folder)
    node_type_file = "node_type_{}.md".format(node_type)
    return os.path.join(error_report_folder, node_type_file)


# get_error_df for each node type
def get_error_dict(remote_version_path, best_model_collection):
    collection = best_model_collection.collection
    error_dict = dict()
    error_dict_with_weight = dict()
    for energy_source in collection.keys():
        error_dict[energy_source] = dict()
        error_dict_with_weight[energy_source] = dict()
        for output_type_name in collection[energy_source].keys():
            items = []
            weight_items = []
            for feature_group_name, best_item in collection[energy_source][output_type_name].items():
                best_item_with_weight = best_model_collection.get_best_item_with_weight(energy_source, output_type_name, feature_group_name)
                if best_item is not None:
                    items += [{"Feature group": feature_group_name, "Model name": best_item.model_name, "MAE": "{:.2f}".format(best_item.metadata["mae"]), "MAPE (%)": "{:.1f}".format(best_item.metadata["mape"]), "URL": best_item.get_archived_filepath(remote_version_path)}]
                if best_item_with_weight is not None:
                    weight_items += [
                        {"Feature group": feature_group_name, "Model name": best_item_with_weight.model_name, "MAE": "{:.2f}".format(best_item_with_weight.metadata["mae"]), "MAPE (%)": "{:.1f}".format(best_item_with_weight.metadata["mape"]), "URL": best_item_with_weight.get_weight_filepath(remote_version_path)}
                    ]
            error_dict[energy_source][output_type_name] = pd.DataFrame(items)
            error_dict_with_weight[energy_source][output_type_name] = pd.DataFrame(weight_items)
    return error_dict, error_dict_with_weight


def format_error_report(error_dict):
    content = ""
    for energy_source in sorted(error_dict.keys()):
        for outputy_type_name in sorted(error_dict[energy_source].keys()):
            df = error_dict[energy_source][outputy_type_name]
            content += "### {} {} model\n\n".format(energy_source, outputy_type_name)
            if len(df) == 0:
                content += "No model available\n\n"
            else:
                content += data_to_markdown_table(df.sort_values(by=["Feature group"]))
    return content


# generate_report_results - version/pipeline_name/error_report/node_type_x.md
def generate_report_results(local_export_path, best_model_collections, node_type_index_json, remote_version_path):
    for node_type, collection in best_model_collections.items():
        if best_model_collections[int(node_type)].has_model:
            markdown_filepath = _error_report_url(local_export_path, node_type, assure=True)
            error_dict, error_dict_with_weight = get_error_dict(remote_version_path, collection)
            markdown_content = "# Validation results on node type {}\n\n".format(node_type)
            markdown_content += data_to_markdown_table(pd.DataFrame([node_type_index_json[str(node_type)]["attrs"]])) + "\n"

            # add links
            markdown_content += "[With local estimator](#with-local-estimator)\n\n"
            markdown_content += "[With sidecar estimator](#with-sidecar-estimator)\n\n"
            # add content
            markdown_content += "## With local estimator\n\n"
            markdown_content += format_error_report(error_dict_with_weight)
            markdown_content += "## With sidecar estimator\n\n"
            markdown_content += format_error_report(error_dict)
            write_markdown(markdown_filepath, markdown_content)


# generate_pipeline_readme - version/pipeline_name/README.md
def generate_pipeline_readme(pipeline_name, local_export_path, node_type_index_json, best_model_collections):
    markdown_filepath = os.path.join(local_export_path, "README.md")
    markdown_content = "# {} on v{} Build\n\n".format(pipeline_name, version)
    markdown_content += "MAE Threshold = {}, MAPE Threshold = {}%\n\n".format(mae_threshold, int(mape_threshold))
    items = []
    for node_type, spec_json in node_type_index_json.items():
        if best_model_collections[int(node_type)].has_model:
            error_file = _error_report_url(".", node_type, assure=False)
            item = {"node type": node_type}
            item.update(spec_json["attrs"])
            item["member size"] = len(spec_json["members"])
            item["error report"] = "[link]({})".format(error_file)
            items += [item]
    df = pd.DataFrame(items)
    markdown_content += "Available Node Type: {}\n\n".format(len(df))
    # add metadata figures
    for ot in ModelOutputType:
        for energy_source in PowerSourceMap.keys():
            data_filename = _pipeline_model_metadata_filename(energy_source, ot.name)
            markdown_content += "![]({}.png)\n".format(data_filename)

    markdown_content += data_to_markdown_table(df.sort_values(by=["node type"]))
    # add power curve figures
    for ot in ModelOutputType:
        for energy_source in PowerSourceMap.keys():
            data_filename = _power_curve_filename(energy_source, ot.name)
            png_filename = data_filename + ".png"
            markdown_content += "## {} ({})\n".format(energy_source, ot.name)
            markdown_content += "![]({})\n".format(png_filename)

    write_markdown(markdown_filepath, markdown_content)
    return markdown_filepath


# append_version_readme - version/README.md
def append_version_readme(local_version_path, pipeline_metadata):
    readme_path = os.path.join(local_version_path, "README.md")
    content_to_append = "[{0}](./.doc/{0}.md)|{1}|{2}|[{3}](https://github.com/{3})|[link](./{0})\n".format(pipeline_metadata["name"], pipeline_metadata["collect_time"], pipeline_metadata["last_update_time"], pipeline_metadata["publisher"])
    with open(readme_path, "a") as file:
        file.write(content_to_append)

