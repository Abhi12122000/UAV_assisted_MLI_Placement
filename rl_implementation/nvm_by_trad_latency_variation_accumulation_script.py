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
system_data_folder = os.path.join(current_dir, "dataset")
system_data_filename = "system_data.json"

latency_multiplier_values = np.arange(1, 10.5, 1)
accumulation_dict = {}

#initializing system_data and global_user_data by running the script once
print(f"initializing data, running main")
main_file.main(algo_type="spa", copy_env_from_file=False)
with open(os.path.join(system_data_folder, system_data_filename), "r") as system_data_file:
    original_system_data = json.load(system_data_file)

with open(os.path.join(system_data_folder, system_data_filename), "r") as system_data_file:
    system_data = json.load(system_data_file)
    
copy_env_from_file = True
for algo_type in ["spa", "dts", "improved_dts"]:
    for latency_multiplier in latency_multiplier_values:
        for server in system_data["server_mli_computation_coeff"]:
            if server == "cloud":
                continue
            for mli_idx in range(len(system_data["server_mli_computation_coeff"][server]["trad_memory"])):
                system_data["server_mli_computation_coeff"][server]["nvm_memory"][mli_idx] = \
                    (system_data["server_mli_computation_coeff"][server]["trad_memory"][mli_idx] * float(latency_multiplier))
            
        with open(os.path.join(system_data_folder, system_data_filename), "w") as system_data_outfile:
            json.dump(system_data, system_data_outfile, indent=4)
        print(f"for {latency_multiplier=}, running main")
        main_file.main(algo_type=algo_type, copy_env_from_file=copy_env_from_file)
        with open(os.path.join(matching_results_folder_path, algo_type + matching_results_dict_filename), "r") as result_file:
            curr_result_dict = json.load(result_file)
        curr_profit = curr_result_dict["profit"][algo_type + "_profit"]
        if algo_type not in accumulation_dict:
            accumulation_dict[algo_type] = {}
        accumulation_dict[algo_type][float(latency_multiplier)] = curr_profit

        if algo_type != "spa":
            continue
        testing_gurobi.apply_gurobi(algo_type=algo_type)
        with open(os.path.join(matching_results_folder_path, "optimalmatching_results.json"), "r") as optimal_result_file:
            optimal_result_dict = json.load(optimal_result_file)
        curr_optimal_profit = optimal_result_dict["profit"]["optimal_profit"]
        if "optimal" not in accumulation_dict:
            accumulation_dict["optimal"] = {}
        if float(latency_multiplier) in accumulation_dict["optimal"]:
            curr_optimal_profit = max(curr_optimal_profit, accumulation_dict["optimal"][float(latency_multiplier)])
        accumulation_dict["optimal"][float(latency_multiplier)] = curr_optimal_profit

with open(os.path.join(system_data_folder, system_data_filename), "w") as system_data_outfile:
    json.dump(original_system_data, system_data_outfile, indent=4)

accumulation_dict_save_filename = "accumulated_data.json"
accumulation_dict_save_folder = os.path.join(current_dir, "comparative_plots", "nvm_by_trad_latency_comparison_plot_folder")
os.makedirs(accumulation_dict_save_folder, exist_ok = True)
with open(os.path.join(accumulation_dict_save_folder, accumulation_dict_save_filename), "w") as outfile:
    json.dump(accumulation_dict, outfile, indent=4)