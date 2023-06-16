import os
# from ..memory_config_fraction.accumulation_script import garbage_func
import main_file
import numpy as np
import json
import testing_gurobi
# garbage_func()
current_dir = os.getcwd()
print(current_dir)
matching_results_folder_path = os.path.join(current_dir, "dataset", "local_dataset")
matching_results_dict_filename = "matching_results.json"

bandwidth_multiplier_values = np.arange(0, 20, 2)    # till 20
accumulation_dict = {}

#initializing system_data and global_user_data by running the script once
main_file.main(algo_type="spa", copy_env_from_file=False)
copy_env_from_file = True
for algo_type in ["spa", "dts", "improved_dts"]:
    for fs_bandwidth_plot_multiplier in bandwidth_multiplier_values:
        
        if fs_bandwidth_plot_multiplier == 0:
            fs_bandwidth_plot_divisor = 1 / 0.0001
        else:
            fs_bandwidth_plot_divisor = 1 / fs_bandwidth_plot_multiplier
        main_file.main(algo_type=algo_type, copy_env_from_file=copy_env_from_file, fs_bandwidth_plot_divisor=fs_bandwidth_plot_divisor)
        with open(os.path.join(matching_results_folder_path, algo_type + matching_results_dict_filename), "r") as result_file:
            curr_result_dict = json.load(result_file)
        curr_profit = curr_result_dict["profit"][algo_type + "_profit"]
        if algo_type not in accumulation_dict:
            accumulation_dict[algo_type] = {}
        accumulation_dict[algo_type][float(fs_bandwidth_plot_multiplier)] = curr_profit

        if algo_type != "spa":
            continue
        testing_gurobi.apply_gurobi(algo_type=algo_type)
        with open(os.path.join(matching_results_folder_path, "optimalmatching_results.json"), "r") as optimal_result_file:
            optimal_result_dict = json.load(optimal_result_file)
        curr_optimal_profit = optimal_result_dict["profit"]["optimal_profit"]
        if "optimal" not in accumulation_dict:
            accumulation_dict["optimal"] = {}
        if float(fs_bandwidth_plot_multiplier) in accumulation_dict["optimal"]:
            curr_optimal_profit = max(curr_optimal_profit, accumulation_dict["optimal"][float(fs_bandwidth_plot_multiplier)])
        accumulation_dict["optimal"][float(fs_bandwidth_plot_multiplier)] = curr_optimal_profit
        

accumulation_dict_save_filename = "accumulated_data.json"
accumulation_dict_save_folder = os.path.join(current_dir, "comparative_plots", "bandwidth_comparison_plot_folder")
with open(os.path.join(accumulation_dict_save_folder, accumulation_dict_save_filename), "w") as outfile:
    json.dump(accumulation_dict, outfile, indent=4)