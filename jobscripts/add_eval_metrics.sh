#!/bin/bash

# Best-performing model only
# SBCSAE
## flan-t5-xl
### "Complete this sentence: "
python3 add_eval_metrics.py -inp "exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_cts_fine-tune_chrf.json" -out "exp/completion/SBCSAE/flan-t5-xl_cts_fine-tune_chrf"

