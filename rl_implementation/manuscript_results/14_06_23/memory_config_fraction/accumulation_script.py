import json
import os

current_dir = os.getcwd()
local_dataset_directory = os.path.join(current_dir, 'dataset', 'local_dataset')
memory_config_fraction_directory = os.path.join(current_dir, "comparative_plots", "memory_config_fraction")

with open(os.path.join(local_dataset_directory, "matching_results.json")) as infile:
    profit_matching_dict = json.load(infile)

accumulated_points_dict = {'points': []}
if "accumulated_points_dict.json" in os.listdir(memory_config_fraction_directory):
    with open(os.path.join(memory_config_fraction_directory, "accumulated_points_dict.json")) as infile:
        accumulated_points_dict = json.load(infile)

current_dict = {
        'profit': profit_matching_dict['profit'], 
        'UE_count': profit_matching_dict['UE_count']
    }

if "normalizing_profit" in profit_matching_dict:
    current_dict["normalizing_profit"] = profit_matching_dict["normalizing_profit"]
accumulated_points_dict['points'].append(current_dict)

with open(os.path.join(memory_config_fraction_directory, "accumulated_points_dict.json"), "w") as outfile:
    json.dump(accumulated_points_dict, outfile, indent=4)


def garbage_func():
    print("inside garbage func")