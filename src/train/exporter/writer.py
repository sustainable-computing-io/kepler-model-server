import os
import sys

import pandas as pd

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

from loader import load_json, get_machine_path, default_init_model_url, get_url, trainers_with_weight
from config import ERROR_KEY
from train_types import ModelOutputType, FeatureGroup

def write_markdown(markdown_filepath, markdown_content):

    try:
        with open(markdown_filepath, "w", encoding="utf-8") as markdown_file:
            # Write the Markdown content to the file
            markdown_file.write(markdown_content)
            print(f"Markdown file '{markdown_filepath}' has been created successfully.")
    except IOError as e:
        print(f"Cannot write '{markdown_filepath}': {e}")


# Function to convert a dictionary to a Markdown table
def dict_to_markdown_table(data):
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
    content = dict_to_markdown_table(df)
    content += "\nrepetition: {}".format(spec["repetition"])
    return content

def format_trainer(trainers):
    trainer_content = ""
    for trainer in trainers.split(","):
        trainer_content += "  - {}\n".format(trainer)
    return trainer_content

def _version_path(machine_path):
    return os.path.join(machine_path, "..")

def generate_pipeline_page(data_path, machine_path, train_args, skip_if_exist=True):
    doc_path = os.path.join(_version_path(machine_path), ".doc")
    pipeline_name = train_args["pipeline_name"]
    markdown_filename = "{}.md".format(pipeline_name)
    markdown_filepath = os.path.join(doc_path, markdown_filename)
    if skip_if_exist and os.path.exists(markdown_filepath):
        print(f"Markdown file '{markdown_filepath}' already exists.")
        return

    workload_content = ""
    inputs = train_args["input"].split(",")
    for input in inputs:
        benchmark_name = "".join(input.split("_")[0:-2])
        data = load_json(data_path, benchmark_name)

        workload_content += """
### {}

<-- put workload description here -->

<details>

{}

</details>

        """.format(benchmark_name, format_cpe_content(data))


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
    """.format(pipeline_name, train_args["extractor"], train_args["isolator"], format_trainer(train_args["abs_trainers"]), "  (same as AbsPower Trainers)" if train_args["abs_trainers"] == train_args["dyn_trainers"] else train_args["dyn_trainers"], workload_content)

    write_markdown(markdown_filepath, markdown_content)

def model_url(version, machine_id, pipeline_name, energy_source, output_type, feature_group, model_name, weight):
    machine_path = get_machine_path(default_init_model_url, version, machine_id, assure=False)
    model_url = get_url(ModelOutputType[output_type], FeatureGroup[feature_group], model_name=model_name, model_topurl=machine_path, energy_source=energy_source, pipeline_name=pipeline_name, weight=weight)
    return model_url

def format_error_content(train_args, mae_validated_df_map, weight):
    content = ""
    for energy_source, mae_validated_df_outputs in mae_validated_df_map.items():
        for output_type, mae_validated_df in mae_validated_df_outputs.items():
            content += "### {} {} model\n\n".format(energy_source, output_type)
            df = mae_validated_df
            if weight:
                df = mae_validated_df[mae_validated_df["model_name"].str.contains('|'.join(trainers_with_weight))]
            items = []
            min_err_rows = df.loc[df.groupby(["feature_group"])[ERROR_KEY].idxmin()]
            for _, row in min_err_rows.iterrows():
                item = dict()
                feature_group = row["feature_group"]
                model_name = row["model_name"]
                item["url"] = model_url(train_args["version"], train_args["machine_id"], train_args["pipeline_name"], energy_source, output_type, feature_group, model_name, weight)
                item[ERROR_KEY] = "{:.2f}".format(row[ERROR_KEY])
                item["feature group"] = feature_group
                item["model name"] = model_name
                items += [item]
            print_df = pd.DataFrame(items, columns=["feature group", "model name", ERROR_KEY, "url"])
            content += dict_to_markdown_table(print_df.sort_values(by=["feature group"]))
    return content

def generate_validation_results(machine_path, train_args, mae_validated_df_map):
    markdown_filepath = os.path.join(machine_path, "README.md")

    markdown_content = "# Validation results\n\n"
    markdown_content += "## With local estimator\n\n"
    markdown_content += format_error_content(train_args, mae_validated_df_map, weight=True)
    markdown_content += "## With sidecar estimator\n\n"
    markdown_content += format_error_content(train_args, mae_validated_df_map, weight=False)
    write_markdown(markdown_filepath, markdown_content)

def append_version_readme(machine_path, train_args, pipeline_metadata, include_raw):
    readme_path = os.path.join(_version_path(machine_path), "README.md")

    content_to_append = "{0}|[{1}](./.doc/{1}.md)|{2}|{3}|{4}|[{5}](https://github.com/{5})|[link](./{6}/README.md)\n".format(train_args["machine_id"],  \
           train_args["pipeline_name"], \
           "&check;" if include_raw else "X", \
           pipeline_metadata["collect_time"], \
           pipeline_metadata["last_update_time"], \
           pipeline_metadata["publisher"],\
           train_args["machine_id"]\
           )

    with open(readme_path, 'a') as file:
        file.write(content_to_append)