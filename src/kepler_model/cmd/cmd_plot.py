import os
from kepler_model.util.prom_types import TIMESTAMP_COL
from kepler_model.util import PowerSourceMap

from kepler_model.util.train_types import FeatureGroup, ModelOutputType, weight_support_trainers
from kepler_model.util.loader import load_metadata, load_scaler, get_model_group_path
from kepler_model.train.profiler.node_type_index import NodeTypeIndexCollection
from kepler_model.estimate import load_model

markers = ["o", "s", "^", "v", "<", ">", "p", "P", "*", "x", "+", "|", "_"]


def ts_plot(data, cols, title, output_folder, name, labels=None, subtitles=None, ylabel=None):
    plot_height = 3
    plot_width = 10
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(font_scale=1.2)
    fig, axes = plt.subplots(len(cols), 1, figsize=(plot_width, len(cols) * plot_height))
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


def feature_power_plot(data, model_id, output_type, energy_source, feature_cols, actual_power_cols, predicted_power_cols, output_folder, name):
    plot_height = 5
    plot_width = 5

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(font_scale=1.2)
    row_num = len(feature_cols)
    col_num = len(actual_power_cols)
    width = max(10, col_num * plot_width)
    fig, axes = plt.subplots(row_num, col_num, figsize=(width, row_num * plot_height))
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
            sns.lineplot(data=sorted_data, x=feature_col, y=predicted_power_cols[yi], ax=ax, label="predicted", color="C1")
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


def summary_plot(args, energy_source, summary_df, output_folder, name):
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
    fig, axes = plt.subplots(col_num, 1, figsize=(plot_width, plot_height * col_num))
    for i in range(0, col_num):
        component = energy_components[i]
        data = summary_df[(summary_df["energy_source"] == energy_source) & (summary_df["energy_component"] == component)]
        data = data.sort_values(by=["Feature Group", "MAE"])
        if col_num == 1:
            ax = axes
        else:
            ax = axes[i]
        sns.barplot(data=data, x="Feature Group", y="MAE", hue="Model", ax=ax)
        ax.set_title(component)
        ax.set_ylabel("MAE (Watt)")
        ax.set_ylim((0, 100))
        if i < col_num - 1:
            ax.set_xlabel("")
        ax.legend(bbox_to_anchor=(1.05, 1.05))
    plt.suptitle("{} {} error".format(energy_source, args.output_type))
    plt.tight_layout()
    filename = os.path.join(output_folder, name + ".png")
    fig.savefig(filename)
    plt.close()


def metadata_plot(args, energy_source, metadata_df, output_folder, name):
    if metadata_df is None or len(metadata_df) == 0:
        print("no metadata data to plot")
        return

    plot_height = 5
    plot_width = 20

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(font_scale=1.2)

    energy_components = PowerSourceMap[energy_source]
    col_num = len(energy_components)
    fig, axes = plt.subplots(col_num, 1, figsize=(plot_width, plot_height * col_num))
    for i in range(0, col_num):
        component = energy_components[i]
        metadata_df = metadata_df.sort_values(by="feature_group")
        if col_num == 1:
            ax = axes
        else:
            ax = axes[i]
        sns.barplot(data=metadata_df, x="feature_group", y="mae", hue="trainer", ax=ax, hue_order=sorted(metadata_df["trainer"].unique()), errorbar=None, palette="Set3")
        ax.set_title(component)
        ax.set_ylabel("MAE (Watt)")
        ax.set_xlabel("Feature Group")
        # ax.set_ylim((0, 100))
        if i < col_num - 1:
            ax.set_xlabel("")
    #  ax.legend(bbox_to_anchor=(1.05, 1.05))
    plt.suptitle("Pipieline metadata of {} {}".format(energy_source.upper(), args.output_type))
    plt.tight_layout()
    plt.legend(frameon=False)
    filename = os.path.join(output_folder, name + ".png")
    fig.savefig(filename)
    plt.close()


def power_curve_plot(args, data_path, energy_source, output_folder, name):
    if "MODEL_PATH" in os.environ:
        model_toppath = os.environ["MODEL_PATH"]
    else:
        model_toppath = data_path
    pipeline_name = args.pipeline_name
    pipeline_path = os.path.join(model_toppath, pipeline_name)
    node_collection = NodeTypeIndexCollection(pipeline_path)
    all_node_types = sorted(list(node_collection.node_type_index.keys()))
    output_type = ModelOutputType[args.output_type]
    models, _, cpu_ms_max = _load_all_models(model_toppath=model_toppath, output_type=output_type, name=pipeline_name, node_types=all_node_types, energy_source=energy_source)
    if len(models) > 0:
        _plot_models(models, cpu_ms_max, energy_source, output_folder, name)


def _get_model(model_toppath, trainer, model_node_type, output_type, name, energy_source):
    feature_group = FeatureGroup.BPFOnly
    model_name = "{}_{}".format(trainer, model_node_type)
    group_path = get_model_group_path(model_toppath, output_type, feature_group, energy_source, name)
    model_path = os.path.join(group_path, model_name)
    model = load_model(model_path)
    metadata = load_metadata(model_path)
    if metadata is None:
        return model, None, None
    scaler = load_scaler(model_path)
    cpu_ms_max = scaler.max_abs_[0]
    return model, metadata, cpu_ms_max


def _load_all_models(model_toppath, output_type, name, node_types, energy_source):
    models_dict = dict()
    metadata_dict = dict()
    cpu_ms_max_dict = dict()
    for model_node_type in node_types:
        min_mae = None
        for trainer in weight_support_trainers:
            model, metadata, cpu_ms_max = _get_model(model_toppath, trainer, model_node_type, output_type=output_type, name=name, energy_source=energy_source)
            if metadata is None:
                continue
            cpu_ms_max_dict[model_node_type] = cpu_ms_max
            if min_mae is None or min_mae > metadata["mae"]:
                min_mae = metadata["mae"]
                models_dict[model_node_type], metadata_dict[model_node_type] = model, metadata
    return models_dict, metadata_dict, cpu_ms_max_dict


def _plot_models(models, cpu_ms_max, energy_source, output_folder, name, max_plot=15, cpu_time_bin_num=10, sample_num=20):
    from kepler_model.util.train_types import BPF_FEATURES
    import numpy as np
    import pandas as pd
    import seaborn as sns

    sns.set_palette("Paired")

    import matplotlib.pyplot as plt

    main_feature_col = BPF_FEATURES[0]
    predicted_col = {"acpi": "default_platform_power", "rapl-sysfs": "default_package_power"}

    num_bins = len(cpu_ms_max) // cpu_time_bin_num + 1
    nobin = False
    if num_bins == 1:
        nobin = True
    values = np.array(list(cpu_ms_max.values()))
    _, bins = np.histogram(values, bins=num_bins)
    bin_size = len(bins) + 1 if not nobin else 1
    data_with_prediction_list = [[] for _ in range(bin_size)]

    num_cols = min(3, bin_size)

    for node_type, model in models.items():
        # generate data from scaler
        xs = np.column_stack((np.linspace(0, cpu_ms_max[node_type], sample_num), np.zeros(sample_num)))
        data = pd.DataFrame(xs, columns=models[node_type].estimator.features)
        _, data_with_prediction = model.append_prediction(data)
        if nobin:
            bin_index = 0
        else:
            bin_index = np.digitize([cpu_ms_max[node_type]], bins)[0]
        data_with_prediction_list[bin_index] += [(node_type, data_with_prediction)]
    total_graphs = 0
    for data_with_predictions in data_with_prediction_list:
        total_graphs += int(np.ceil(len(data_with_predictions) / max_plot))
    num_rows = int(np.ceil(total_graphs / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(int(6 * num_cols), int(5 * num_rows)))
    axes_index = 0
    for data_with_predictions in data_with_prediction_list:
        index = 0
        for data_with_prediction_index in data_with_predictions:
            if num_rows == 1 and num_cols == 1:
                ax = axes
            else:
                ax = axes[axes_index // num_cols][axes_index % num_cols]
            node_type = data_with_prediction_index[0]
            data_with_prediction = data_with_prediction_index[1]
            sns.lineplot(data=data_with_prediction, x=main_feature_col, y=predicted_col[energy_source], label="type={}".format(node_type), marker=markers[index], ax=ax)
            index += 1
            index = index % len(markers)
            if index % max_plot == 0:
                ax.set_ylabel("Predicted power (W)")
                axes_index += 1
        if len(data_with_predictions) > 0:
            ax.set_ylabel("Predicted power (W)")
            axes_index += 1
    filename = os.path.join(output_folder, name + ".png")
    plt.tight_layout()
    fig.savefig(filename)
    plt.close()
