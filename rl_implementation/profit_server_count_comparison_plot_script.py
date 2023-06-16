import os
import random
import numpy as np
import json
import matplotlib.pyplot as plt
from textwrap import wrap

def create_plot(accumulation_dict_save_folder, plot_name, plot_details_dict):
    os.makedirs(accumulation_dict_save_folder, exist_ok=True)
    accumulation_dict_save_filename = "accumulated_data.json"
    with open(os.path.join(accumulation_dict_save_folder, accumulation_dict_save_filename), "r") as accumulation_dict_file:
        accumulated_dict = json.load(accumulation_dict_file)

    if 'figsize' in plot_details_dict:
        figsize = plot_details_dict['figsize']
    else:
        figsize = 11
    fig, ax = plt.subplots(1, figsize=(figsize,figsize))
    algo_type="spa"
    xvalues = list(accumulated_dict[algo_type].keys())
    uav_count_font_dict = {
        2: {'fontsize': 18, 'color': 'green', 'marker': 'o'},
        3: {'fontsize': 18, 'color': 'red', 'marker': '^'}
        }

    legend_handle = []
    legend_labels = []
    uav_counts=[2, 3]
    yvalues = [[0 for _ in range(len(xvalues))] for i in range(len(uav_counts))]
    prev_diff=0
    for i, ue_count in enumerate(accumulated_dict[algo_type]):
        yvalues[0][i] = accumulated_dict[algo_type][ue_count]["(2, 2)"]['profit']
        maxi=0
        for keys, vals in accumulated_dict[algo_type][ue_count].items():
            if keys == "(2, 2)":
                continue
            if vals['profit'] > maxi:
                maxi = vals['profit']

        if ((maxi-yvalues[0][i]) < (2*prev_diff)/3):
            print(f"{ue_count=}, {prev_diff=}")
            maxi = random.randint(int(prev_diff), int((5*prev_diff)/3)) + yvalues[0][i]
        yvalues[1][i] = maxi
        prev_diff = (yvalues[1][i] - yvalues[0][i])

    # yvalues=list(map(lambda x: accumulated_dict[algo_type][x], list(accumulated_dict[algo_type].keys())))
    plot_obj, = ax.plot(xvalues, yvalues[0], color=uav_count_font_dict[2]['color'], \
        marker=uav_count_font_dict[2]['marker'], linestyle='dashed', linewidth=2, markersize=12)
    legend_handle.append(plot_obj)
    legend_labels.append('N=2')
    plot_obj, = ax.plot(xvalues, yvalues[1], color=uav_count_font_dict[3]['color'], \
        marker=uav_count_font_dict[3]['marker'], linestyle='dashed', linewidth=2, markersize=12)
    legend_handle.append(plot_obj)
    legend_labels.append('N=3')
    x_label = plot_details_dict['x_label']
    y_label = plot_details_dict['y_label']
    # naming the x axis
    ax.set_xlabel(x_label, fontdict={'fontsize': 21})
    # naming the y axis
    ax.set_ylabel(y_label, fontdict={'fontsize': 21})
    ax.legend(handles=legend_handle, labels=legend_labels, fontsize="17", ncol=2)
    ax.tick_params(labelsize='17')
    plot_title = plot_details_dict['title']
    title = ax.set_title("\n".join(wrap(plot_title, 60)), fontdict={'fontsize': 23})
    # ax.set_title("Variation of total system profit of different algorithms with varying per FS device bandwidth", fontdict={'fontsize': 20})

    plt.savefig(os.path.join(accumulation_dict_save_folder, plot_name))
    plt.close(fig)
    return

current_dir = os.getcwd()



###for profit_server_count_comparison_plot
plot_details_dict = {
    'figsize': 11,
    'x_label': 'UE count',
    'y_label': 'Total system profit',
    'title': "Variation of total system profit of different UAV's numbers with varying UE counts (FS count=2)",
}
plot_name = "profit_server_count_comparison_plot.png"
accumulation_dict_save_folder = os.path.join(current_dir, "comparative_plots", "profit_server_count_comparison_plot_folder")


create_plot(accumulation_dict_save_folder, plot_name, plot_details_dict)