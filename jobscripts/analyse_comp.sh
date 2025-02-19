#!/bin/bash

# SBCSAE
## flan-t5-xl
### Without prefix
#python3 analyse_completion.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_nopx_fine-tune_chrf.json" > exp/completion/SBCSAE/flan-t5-xl/analyse_nopx_flan-t5-xl.txt
### "Complete this sentence: "
python3 analyse_completion.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_cts_fine-tune_chrf.json" > exp/completion/SBCSAE/flan-t5-xl/analyse_cts_flan-t5-xl.txt

## t5-large
### Without prefix
#python3 analyse_completion.py -inp "exp/completion/SBCSAE/t5-large/t5-large_nopx_fine-tune_chrf.json" > exp/completion/SBCSAE/t5-large/analyse_nopx_t5-large.txt
### "Complete this sentence: "
#python3 analyse_completion.py -inp "exp/completion/SBCSAE/t5-large/t5-large_cts_fine-tune_chrf.json" > exp/completion/SBCSAE/t5-large/analyse_cts_t5-large.txt

## flan-t5-base
### Without prefix
#python3 analyse_completion.py -inp "exp/completion/SBCSAE/flan-t5-base/flan-t5-base_nopx_fine-tune_chrf.json" > exp/completion/SBCSAE/flan-t5-base/analyse_nopx_flan-t5-base.txt
### "Complete this sentence: "
#python3 analyse_completion.py -inp "exp/completion/SBCSAE/flan-t5-base/flan-t5-base_cts_fine-tune_chrf.json" > exp/completion/SBCSAE/flan-t5-base/analyse_cts_flan-t5-base.txt

## t5-base
### Without prefix
#python3 analyse_completion.py -inp "exp/completion/SBCSAE/t5-base/t5-base_nopx_fine-tune_chrf.json" > exp/completion/SBCSAE/t5-base/analyse_nopx_t5-base.txt
### "Complete this sentence: "
#python3 analyse_completion.py -inp "exp/completion/SBCSAE/t5-base/t5-base_cts_fine-tune_chrf.json" > exp/completion/SBCSAE/t5-base/analyse_cts_t5-base.txt
