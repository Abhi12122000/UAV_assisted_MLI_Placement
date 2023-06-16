import os
# from ..memory_config_fraction.accumulation_script import garbage_func
import main_file
import numpy as np
import json
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
server_count_pairs = [(1, 1), (2, 2), (3, 3), (4, 4)]
# #TESTING CODE
# server_count_pairs = [(1, 3)]
# #TESTING CODE BLOCK ENDS
accumulation_dict = {}

copy_env_from_file = True
algo_type = "spa"
for ue_count in [i for i in range(100, 700, 100)]:
    for server_count in server_count_pairs:
        reassign_server_counts_dict = {
            "UAV_count": server_count[0],
            "UE_count": ue_count,
            "num_UE_centers": server_count[0],
            "FS_count": server_count[1]
        }
        for key in reassign_server_counts_dict:
            env_params[key] = reassign_server_counts_dict[key]
        with open(os.path.join(env_params_folder, env_params_filename), "w") as env_params_outfile:
            json.dump(env_params, env_params_outfile, indent=4)
        
        print(f"for {ue_count=}, {server_count=}, running main")
        main_file.main(algo_type=algo_type, copy_env_from_file=copy_env_from_file, reassign_server_counts=reassign_server_counts_dict)
        with open(os.path.join(matching_results_folder_path, algo_type + matching_results_dict_filename), "r") as result_file:
            curr_result_dict = json.load(result_file)
        curr_profit = curr_result_dict["profit"]["spa_profit"]
        if algo_type not in accumulation_dict:
            accumulation_dict[algo_type] = {}
        if ue_count not in accumulation_dict[algo_type]:
            accumulation_dict[algo_type][ue_count] = {}
        accumulation_dict[algo_type][ue_count][str(server_count)] = {'profit': curr_profit}

# #use this line to restore original system_data.json when increasing UE_count functionality is added
# with open(os.path.join(system_data_folder, system_data_filename), "w") as system_data_outfile:
#     json.dump(system_data, system_data_outfile, indent=4)

accumulation_dict_save_filename = "accumulated_data.json"
accumulation_dict_save_folder = os.path.join(current_dir, "comparative_plots", "profit_server_count_comparison_plot_folder")
os.makedirs(accumulation_dict_save_folder, exist_ok = True)
with open(os.path.join(accumulation_dict_save_folder, accumulation_dict_save_filename), "w") as outfile:
    json.dump(accumulation_dict, outfile, indent=4)