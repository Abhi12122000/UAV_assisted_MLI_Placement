import os
import pandas as pd
import json

def convert_to_csv(meta_data):
    json_filepath = meta_data['input_datapath']
    with open(json_filepath, "r") as input_file:
        input_dict = json.load(input_file)
    
    transformed_dict = {}
    index_keys = [key for key in input_dict["spa"]]
    transformed_dict[meta_data['key_header']] = index_keys
    for algo in input_dict:
        transformed_dict[algo] = []
        for key, value in input_dict[algo].items():
            transformed_dict[algo].append(value)
        
    df = pd.DataFrame(transformed_dict)
    return df


root_rl_dir = os.getcwd()
comparative_plots_dir = os.path.join(root_rl_dir, "comparative_plots")
bandwidth_folder = os.path.join(comparative_plots_dir, "bandwidth_comparison_plot_folder")
nvm_by_trad_folder = os.path.join(comparative_plots_dir, "nvm_by_trad_latency_comparison_plot_folder")
phi_folder = os.path.join(comparative_plots_dir, "phi_comparison_plot_folder")
ue_count_folder = os.path.join(comparative_plots_dir, "ue_count_variation_plot_folder")

meta_data_dict = {
    'bandwidth_plot': 
    {
        'input_datapath': os.path.join(bandwidth_folder, "accumulated_data.json"),
        'sheetname': 'bandwidth_multiplier_variation_plot',
        'key_header': 'bandwidth multiplier values',
    },
    'nvm_by_trad_plot':
    {
        'input_datapath': os.path.join(nvm_by_trad_folder, "accumulated_data.json"),
        'sheetname': 'nvm_by_trad_latency_ratio_variation_plot',
        'key_header': 'ratio of nvm/trad latency on servers',
    },
    'phi_plot':
    {
        'input_datapath': os.path.join(phi_folder, "accumulated_data.json"),
        'sheetname': 'phi_variation_plot',
        'key_header': 'energy consumption cost multiplier values',
    },
    'ue_count_plot':
    {
        'input_datapath': os.path.join(ue_count_folder, "accumulated_data.json"),
        'sheetname': 'ue_count_variation_plot',
        'key_header': 'number of UEs in system',
    },
}

with pd.ExcelWriter(os.path.join(os.getcwd(), 'Abhishek_MTP.xlsx')) as writer:  
    for key, meta_data in meta_data_dict.items(): 
        convert_to_csv(meta_data=meta_data).to_excel(writer, sheet_name=meta_data['sheetname'])
