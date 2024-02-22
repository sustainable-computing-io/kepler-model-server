import os
import sys

cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)

from util.prom_types import TIMESTAMP_COL
from util import PowerSourceMap


def ts_plot(data, cols, title, output_folder, name, labels=None, subtitles=None, ylabel=None):
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

def feature_power_plot(data, model_id, output_type, energy_source, feature_cols, actual_power_cols, predicted_power_cols, output_folder, name):
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
    fig, axes = plt.subplots(col_num, 1, figsize=(plot_width, plot_height*col_num))
    for i in range(0, col_num):
        component = energy_components[i]
        metadata_df = metadata_df.sort_values(by="feature_group")
        if col_num == 1:
            ax = axes
        else:
            ax = axes[i]
        sns.barplot(data=metadata_df, x="feature_group", y="mae", hue="trainer", ax=ax, hue_order=sorted(metadata_df['trainer'].unique()), errorbar=None, palette="Set3")
        ax.set_title(component)
        ax.set_ylabel("MAE (Watt)")
        ax.set_xlabel("Feature Group")
        # ax.set_ylim((0, 100))
        if i < col_num-1:
            ax.set_xlabel("")
       #  ax.legend(bbox_to_anchor=(1.05, 1.05))
    plt.suptitle("Pipieline metadata of {} {}".format(energy_source.upper(), args.output_type))
    plt.tight_layout()
    plt.legend(frameon=False)
    filename = os.path.join(output_folder, name + ".png")
    fig.savefig(filename)
    plt.close()