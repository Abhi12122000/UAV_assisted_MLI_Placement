#!/bin/bash

#run this script from rl_implementation
accumulation_script_path="/Users/abhishekkumar/Documents/MTP_Papers/rl_implementation/comparative_plots/bandwidth_comparison_plot_folder/accumulation_script.py"
plot_script_path="/Users/abhishekkumar/Documents/MTP_Papers/rl_implementation/comparative_plots/bandwidth_comparison_plot_folder/plot_script.py"

python $accumulation_script_path
python $plot_script_path