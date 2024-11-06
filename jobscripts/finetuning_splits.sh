#!/bin/bash

# Experiment folder to take the data from is a command line argument
exp_folder=$1

# SBCSAE
python3 create_finetuning_splits.py -inp exp/rule_base/sbcsae/${exp_folder}/out/syn_data.json -out data/SBCSAE/

# Control
#python3 create_finetuning_splits.py -inp exp/rule_base/control/${exp_folder}/out/syn_data.json -out data/Control/
