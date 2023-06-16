import os
# from ..memory_config_fraction.accumulation_script import garbage_func
# from ..main_file import main as mn
import numpy as np
import json
# garbage_func()
current_dir = os.getcwd()
print(current_dir)
matching_results_folder_path = os.path.join(current_dir, "dataset", "local_dataset")
matching_results_dict_filename = "matching_results.json"

divisor_values = np.arange(0.5, 10, 0.5)
accumulation_dict = {}
for algo_type in ["spa", "dts", "improved_dts"]:
    copy_env_from_file=True
    for fs_bandwidth_plot_divisor in divisor_values:
        mn(algo_type=algo_type, copy_env_from_file=copy_env_from_file, fs_bandwidth_plot_divisor=fs_bandwidth_plot_divisor)
        with open(os.path.join(matching_results_folder_path, algo_type + matching_results_dict_filename), "r") as result_file:
            curr_result_dict = json.load(result_file)
        curr_profit = curr_result_dict["profit"][algo_type + "_profit"]
        if algo_type not in accumulation_dict:
            accumulation_dict[algo_type] = []
        accumulation_dict[algo_type].append({float(fs_bandwidth_plot_divisor): curr_profit})

accumulation_dict_save_filename = "accumulated_data.json"
accumulation_dict_save_folder = os.path.join(current_dir, "comparative_plots", "bandwidth_comparison_plot_folder")
with open(os.path.join(accumulation_dict_save_folder, accumulation_dict_save_filename), "w") as outfile:
    json.dump(accumulation_dict, outfile, indent=4)