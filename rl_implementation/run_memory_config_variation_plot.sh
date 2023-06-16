#!/bin/bash

# data_path="/Users/abhishekkumar/Documents/MTP_Papers/spa_algo_implementation/datasets/generated_data"
### NOTE: run main_file.py only when fraction_testing_mode = True
main_file_path="/Users/abhishekkumar/Documents/MTP_Papers/rl_implementation/main_file.py"
store_memory_config_script_path="/Users/abhishekkumar/Documents/MTP_Papers/rl_implementation/comparative_plots/memory_config_fraction/accumulation_script.py"
# create_memory_fraction_plot_script = "/Users/abhishekkumar/Documents/MTP_Papers/rl_implementation/comparative_plots/memory_config_fraction/memory_config_fraction_plot.py"
data_path="/Users/abhishekkumar/Documents/MTP_Papers/rl_implementation/comparative_plots/memory_config_fraction/accumulated_points_dict.json"

# rm $data_path

# current_directory_size=$(ls $data_path | wc -l) 
i=0
while [ $i -le 10 ];
do
	# set fraction_testing_mode=True in environment
	python $main_file_path
	echo 
	python $store_memory_config_script_path
	i=$((i+1))
done

# python $create_memory_fraction_plot_script

