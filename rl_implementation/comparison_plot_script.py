import os
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
    xvalues = list(accumulated_dict['spa'].keys())
    algo_font_dict = {
        'spa': {'fontsize': 15, 'color': 'green', 'marker': 'o'},
        'dts': {'fontsize': 15, 'color': 'red', 'marker': '^'},
        'improved_dts': {'fontsize': 15, 'color': 'magenta', 'marker': '*'},
        'optimal': {'fontsize': 15, 'color': 'blue', 'marker': 'X'},
        }

    legend_handle = []
    legend_labels = []
    for algo_type in accumulated_dict:
        yvalues=list(map(lambda x: accumulated_dict[algo_type][x], list(accumulated_dict[algo_type].keys())))
        plot_obj, = ax.plot(xvalues, yvalues, color=algo_font_dict[algo_type]['color'], \
            marker=algo_font_dict[algo_type]['marker'], linestyle='dashed', linewidth=2, markersize=12)
        legend_handle.append(plot_obj)
        legend_labels.append(algo_type)

    x_label = plot_details_dict['x_label']
    y_label = plot_details_dict['y_label']
    # naming the x axis
    ax.set_xlabel(x_label, fontdict={'fontsize': 18})
    # naming the y axis
    ax.set_ylabel(y_label, fontdict={'fontsize': 18})
    ax.legend(handles=legend_handle, labels=legend_labels, fontsize="17", ncol=2)
    ax.tick_params(labelsize='13')
    plot_title = plot_details_dict['title']
    title = ax.set_title("\n".join(wrap(plot_title, 60)), fontdict={'fontsize': 20})
    # ax.set_title("Variation of total system profit of different algorithms with varying per FS device bandwidth", fontdict={'fontsize': 20})

    plt.savefig(os.path.join(accumulation_dict_save_folder, plot_name))
    plt.close(fig)
    return

current_dir = os.getcwd()




###for bandwidth_variation_comparison_plot
plot_details_dict = {
    'figsize': 11,
    'x_label': 'Per device bandwidth multiplier for FSs',
    'y_label': 'Total system profit',
    'title': "Variation of total system profit of different algorithms with varying per FS device bandwidth",
}
plot_name = "algo_comparison_on_bandwidth_plot.png"
accumulation_dict_save_folder = os.path.join(current_dir, "comparative_plots", "bandwidth_comparison_plot_folder")


# ###for phi_variation_comparison_plot
# plot_details_dict = {
#     'figsize': 11,
#     'x_label': 'Multiplier for energy consumption cost coefficient (phi)',
#     'y_label': 'Total system profit',
#     'title': "Variation of total system profit of different algorithms with varying energy consumption cost coefficient(phi) multiplier",
# }
# plot_name = "algo_comparison_on_phi_plot.png"
# accumulation_dict_save_folder = os.path.join(current_dir, "comparative_plots", "phi_comparison_plot_folder")


# ###for nvm_by_trad_latency_variation_comparison_plot
# plot_details_dict = {
#     'figsize': 11,
#     'x_label': 'NVM / RAM latency on UAVs and FSs',
#     'y_label': 'Total system profit',
#     'title': "Variation of total system profit of different algorithms with varying NVM / RAM latency in UAVs and FSs",
# }
# plot_name = "algo_comparison_on_nvm_by_trad_latency_plot.png"
# accumulation_dict_save_folder = os.path.join(current_dir, "comparative_plots", "nvm_by_trad_latency_comparison_plot_folder")


# ###for ue_count_variation_plot
# plot_details_dict = {
#     'figsize': 11,
#     'x_label': 'Number of UEs in system',
#     'y_label': 'Total system profit',
#     'title': "Variation of total system profit of different algorithms with varying UE counts in the system (fixed UAV_count=2, FS_count=1)",
# }
# plot_name = "algo_comparison_on_ue_count_plot.png"
# accumulation_dict_save_folder = os.path.join(current_dir, "comparative_plots", "ue_count_variation_plot_folder")


create_plot(accumulation_dict_save_folder, plot_name, plot_details_dict)