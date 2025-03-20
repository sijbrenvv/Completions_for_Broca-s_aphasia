#!/bin/bash

# SBCSAE
## flan-t5-xl
### Without prefix
#python3 negation_dist.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_nopx_fine-tune_chrf.json" > exp/completion/SBCSAE/flan-t5-xl/negation_nopx_flan-t5-xl.txt
### "Complete this sentence: "
python3 negation_dist.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_cts_fine-tune_chrf.json" > exp/completion/SBCSAE/flan-t5-xl/negation_cts_flan-t5-xl.txt

# Authentic
#python3 negation_dist.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_nopx_auth_comp.json" > exp/completion/SBCSAE/flan-t5-xl/negation_auth_nopx_flan-t5-xl.txt
### "Complete this sentence: "
python3 negation_dist.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_cts_auth_comp.json" > exp/completion/SBCSAE/flan-t5-xl/negation_auth_cts_flan-t5-xl.txt
