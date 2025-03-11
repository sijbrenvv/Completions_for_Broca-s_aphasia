#!/bin/bash

# Best-performing model only
# SBCSAE
## flan-t5-xl
### "Complete this sentence: "
python3 add_eval_metrics.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_cts_fine-tune_chrf.json" -out "exp/completion/SBCSAE/flan-t5-xl_cts_fine-tune_chrf"
### Without prefix:
python3 add_eval_metrics.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_nopx_fine-tune_chrf.json" -out "exp/completion/SBCSAE/flan-t5-xl_nopx_fine-tune_chrf"

## t5-large
### "Complete this sentence: "
python3 add_eval_metrics.py -inp "exp/completion/SBCSAE/t5-large/t5-large_cts_fine-tune_chrf.json" -out "exp/completion/SBCSAE/t5-large_cts_fine-tune_chrf"
### Without prefix:
python3 add_eval_metrics.py -inp "exp/completion/SBCSAE/t5-large/t5-large_nopx_fine-tune_chrf.json" -out "exp/completion/SBCSAE/t5-large_nopx_fine-tune_chrf"

## flan-t5-base:
### "Complete this sentence: "
python3 add_eval_metrics.py -inp "exp/completion/SBCSAE/flan-t5-base/flan-t5-base_cts_fine-tune_chrf.json" -out "exp/completion/SBCSAE/flan-t5-base_cts_fine-tune_chrf"
### Without prefix:
python3 add_eval_metrics.py -inp "exp/completion/SBCSAE/flan-t5-base/flan-t5-base_nopx_fine-tune_chrf.json" -out "exp/completion/SBCSAE/flan-t5-base_nopx_fine-tune_chrf"

## t5-base
### "Complete this sentence: "
python3 add_eval_metrics.py -inp "exp/completion/SBCSAE/t5-base/t5-base_cts_fine-tune_chrf.json" -out "exp/completion/SBCSAE/t5-base_cts_fine-tune_chrf"
### Without prefix:
python3 add_eval_metrics.py -inp "exp/completion/SBCSAE/t5-base/t5-base_nopx_fine-tune_chrf.json" -out "exp/completion/SBCSAE/t5-base_nopx_fine-tune_chrf"
