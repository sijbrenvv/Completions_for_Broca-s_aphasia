#!/bin/bash

## Setup data for pre-processing
# Authentic aphasic
python3 setup_ab.py -data "data/Aphasia/all_files/" -out "data/Aphasia/aphasia_broca.csv"
# healthy control data
python3 setup_ab.py -data "data/Control/all_files/" -out "data/Control/control_broca.csv"
# SBCSAE data
python3 setup_sbcsae.py -data "data/SBCSAE/all_files/" -out "data/SBCSAE/sbcsae_broca.csv"

## Pre-process the data
# Authentic aphasic
python3 preprocess_ab.py -inp "data/Aphasia/aphasia_broca.csv" -out "data/Aphasia/aphasia_broca_processed.csv"
# healthy control data
python3 preprocess_ab.py -inp "data/Control/control_broca.csv" -out "data/Control/control_broca_processed.csv"
# SBCSAE data
python3 preprocess_sbcsae.py -inp "data/SBCSAE/sbcsae_broca.csv" -out "data/SBCSAE/sbcsae_broca_processed.csv"