import os
import numpy as np
import json
import matplotlib.pyplot as plt

current_dir = os.getcwd()
accumulation_dict_save_filename = "accumulated_data.json"
accumulation_dict_save_folder = os.path.join(current_dir, "comparative_plots", "bandwidth_comparison_plot_folder")

with open(os.path.join(accumulation_dict_save_folder, accumulation_dict_save_filename), "r") as accumulation_dict_file:
    accumulated_dict = json.load(accumulation_dict_file)

fig_size = 11
fig, ax = plt.subplots(1, figsize=(fig_size,fig_size))
xvalues = list(accumulated_dict['spa'].keys())
algo_font_dict = {
    'spa': {'fontsize': 15, 'color': 'green', 'marker': 'o'},
    'dts': {'fontsize': 15, 'color': 'red', 'marker': '^'}
    }

for algo_type in accumulated_dict:
    yvalues=list(map(lambda x: accumulated_dict[algo_type][x], list(accumulated_dict[algo_type].keys())))
    ax.plot(x=xvalues, y=yvalues, color=algo_font_dict[algo_type]['color'], \
        marker=algo_font_dict[algo_type]['marker'], linestyle='dashed', linewidth=2, markersize=12)

plot_name = "algo_comparison_on_bandwidth_plot.png"
plt.savefig(os.path.join(accumulation_dict_save_folder, plot_name))
plt.close(fig)