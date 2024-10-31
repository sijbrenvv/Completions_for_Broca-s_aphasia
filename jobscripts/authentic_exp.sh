#!/bin/bash

# Experiment folder is fixed
exp_fol="exp/authentic"

# Check if folder already exists, if so, exit, if not, continue
if [ -d "$exp_fol" ]; then
    echo "Experiment folder already exists. Exiting..."
    exit 1
fi

# Set variables
log=${exp_fol}/log/
out=${exp_fol}/out/
mod=${exp_fol}/mod/
bkp=${exp_fol}/bkp/

# Create folders in exp folder
mkdir -p $exp_fol $log $out $mod $bkp

# Define authentic data
authentic_data="data/Aphasia/aphasia_broca_processed.json"
# Define control data
control_data="data/Control/control_broca_processed.json"

# Call create_splits.py to create the splits composed of authentic aphasic and control data
python3 create_splits.py -ad $authentic_data -hd $control_data -out "${exp_fol}/out/" #> ${log}splits
train="${exp_fol}/out/train.json"
dev="${exp_fol}/out/dev.json"

# Train, predict and evaluate
python3 smp_clf/smp_clf_train.py -tr $train -m "NB" -out "${exp_fol}/mod/" #> ${log}train
python3 smp_clf/smp_clf_predict.py -inp $dev -m "${mod}model.pkl" -out "${exp_fol}/out/pred.txt" #> ${log}pred
python3 smp_clf/smp_clf_evaluate.py -t $dev -p "${exp_fol}/out/pred.txt" > ${log}eval.txt

# Backup scripts
cp smp_clf/smp_clf_train.py smp_clf/smp_clf_predict.py smp_clf/smp_clf_evaluate.py create_splits.py $bkp