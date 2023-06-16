import os
# from ..memory_config_fraction.accumulation_script import garbage_func
import main_file
import numpy as np
import json
import testing_gurobi
# garbage_func()
current_dir = os.getcwd()
print(current_dir)
env_params_folder = os.path.join(current_dir, "environment_class")
env_params_filename = "environment_parameters.json"
with open(os.path.join(env_params_folder, env_params_filename), "r") as env_params_file:
    env_params = json.load(env_params_file)

reassign_server_counts_dict = {
                    "UAV_count": 1,
                    "UE_count": 100,
                    "num_UE_centers": 1,
                    "FS_count": 1
                }
for key in reassign_server_counts_dict:
    env_params[key] = reassign_server_counts_dict[key]
with open(os.path.join(env_params_folder, env_params_filename), "w") as env_params_outfile:
    json.dump(env_params, env_params_outfile, indent=4)
# # initializing system_data and global_user_data by running the script once
print(f"initializing data, running main")
main_file.main(algo_type="spa", copy_env_from_file=False)

matching_results_folder_path = os.path.join(current_dir, "dataset", "local_dataset")
matching_results_dict_filename = "matching_results.json"

# server_count_pairs = [(1, 1), (2, 1), (2, 2), (3, 2), (4, 2), (5, 3)]    # (uav_count, fs_count)
# server_count_pairs_for_ues = {
#     100: (2, 1), 
#     200: (2, 2), 
#     300: (3, 2), 
#     400: (4, 3),
#     500: (5, 4),
#     600: (6, 5),
#     700: (7, 6)
# }
server_count_pairs_for_ues = {
    100: (2, 1), 
    200: (2, 1), 
    300: (2, 2), 
    400: (2, 1),
    500: (2, 1),
    600: (2, 1),
    700: (2, 1)
}
accumulation_dict = {}

copy_env_from_file = True
for ue_count in [i for i in range(100, 800, 100)]:
    for algo_type in ["spa", "dts", "improved_dts"]:
        reassign_server_counts_dict = {
            "UAV_count": server_count_pairs_for_ues[ue_count][0],
            "UE_count": ue_count,
            "num_UE_centers": server_count_pairs_for_ues[ue_count][0],
            "FS_count": server_count_pairs_for_ues[ue_count][1]
        }
        for key in reassign_server_counts_dict:
            env_params[key] = reassign_server_counts_dict[key]
        with open(os.path.join(env_params_folder, env_params_filename), "w") as env_params_outfile:
            json.dump(env_params, env_params_outfile, indent=4)
        
        print(f"for {ue_count=}, {reassign_server_counts_dict['UAV_count']=}, running main")
        main_file.main(algo_type=algo_type, copy_env_from_file=copy_env_from_file, reassign_server_counts=reassign_server_counts_dict)
        with open(os.path.join(matching_results_folder_path, algo_type + matching_results_dict_filename), "r") as result_file:
            curr_result_dict = json.load(result_file)
        curr_profit = curr_result_dict["profit"][algo_type + "_profit"]
        if algo_type not in accumulation_dict:
            accumulation_dict[algo_type] = {}
        accumulation_dict[algo_type][ue_count] = curr_profit

        if algo_type != "spa":
            continue
        testing_gurobi.apply_gurobi(algo_type=algo_type)
        with open(os.path.join(matching_results_folder_path, "optimalmatching_results.json"), "r") as optimal_result_file:
            optimal_result_dict = json.load(optimal_result_file)
        curr_optimal_profit = optimal_result_dict["profit"]["optimal_profit"]
        if "optimal" not in accumulation_dict:
            accumulation_dict["optimal"] = {}
        if ue_count in accumulation_dict["optimal"]:
            curr_optimal_profit = max(curr_optimal_profit, accumulation_dict["optimal"][ue_count])
        accumulation_dict["optimal"][ue_count] = curr_optimal_profit
        

# #use this line to restore original system_data.json when increasing UE_count functionality is added
# with open(os.path.join(system_data_folder, system_data_filename), "w") as system_data_outfile:
#     json.dump(system_data, system_data_outfile, indent=4)

accumulation_dict_save_filename = "accumulated_data.json"
accumulation_dict_save_folder = os.path.join(current_dir, "comparative_plots", "ue_count_variation_plot_folder")
os.makedirs(accumulation_dict_save_folder, exist_ok = True)
with open(os.path.join(accumulation_dict_save_folder, accumulation_dict_save_filename), "w") as outfile:
    json.dump(accumulation_dict, outfile, indent=4)