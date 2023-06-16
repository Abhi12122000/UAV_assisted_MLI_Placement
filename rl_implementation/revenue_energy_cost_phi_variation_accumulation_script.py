import os
# from ..memory_config_fraction.accumulation_script import garbage_func
import main_file
import numpy as np
import json
# garbage_func()
current_dir = os.getcwd()
print(current_dir)
matching_results_folder_path = os.path.join(current_dir, "dataset", "local_dataset")
matching_results_dict_filename = "matching_results.json"
data_dict_folder_path = os.path.join(current_dir, "spa_module")
data_dict_filename = "data_dict.json"

with open(os.path.join(data_dict_folder_path, data_dict_filename), "r") as data_dict_file:
    data_dict = json.load(data_dict_file)
phi_val = data_dict["cost_conversion_parameters"]["phi"]

phi_multiplier_values = np.arange(0, 10.5, 0.5)
accumulation_dict = {}

#initializing system_data and global_user_data by running the script once
print(f"initializing data, running main")
main_file.main(algo_type="spa", copy_env_from_file=False)
copy_env_from_file = True
algo_type = "spa"
for phi_multiplier in phi_multiplier_values:
    system_data_folder = os.path.join(current_dir, "dataset")
    system_data_filename = "system_data.json"
    with open(os.path.join(system_data_folder, system_data_filename), "r") as system_data_file:
        system_data = json.load(system_data_file)
    system_data["cost_conversion_parameters"]["phi"] = (phi_val * phi_multiplier)
    with open(os.path.join(system_data_folder, system_data_filename), "w") as system_data_outfile:
        json.dump(system_data, system_data_outfile, indent=4)
    print(f"for {phi_multiplier=}, running main")
    main_file.main(algo_type=algo_type, copy_env_from_file=copy_env_from_file)
    with open(os.path.join(matching_results_folder_path, algo_type + matching_results_dict_filename), "r") as result_file:
        curr_result_dict = json.load(result_file)
    curr_revenue = curr_result_dict["profit"]["revenue"]
    curr_energy_cost = curr_result_dict["profit"]["energy_cost"]
    curr_profit = curr_result_dict["profit"]["spa_profit"]
    if algo_type not in accumulation_dict:
        accumulation_dict[algo_type] = {}
    accumulation_dict[algo_type][float(phi_multiplier)] = {'revenue': curr_revenue, 'energy_cost': curr_energy_cost, 'profit': curr_profit}

system_data["cost_conversion_parameters"]["phi"] = phi_val
with open(os.path.join(system_data_folder, system_data_filename), "w") as system_data_outfile:
    json.dump(system_data, system_data_outfile, indent=4)

accumulation_dict_save_filename = "accumulated_data.json"
accumulation_dict_save_folder = os.path.join(current_dir, "comparative_plots", "revenue_energy_cost_phi_comparison_plot_folder")
os.makedirs(accumulation_dict_save_folder, exist_ok = True)
with open(os.path.join(accumulation_dict_save_folder, accumulation_dict_save_filename), "w") as outfile:
    json.dump(accumulation_dict, outfile, indent=4)