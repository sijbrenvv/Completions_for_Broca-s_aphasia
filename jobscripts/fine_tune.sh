#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition=gpu
#SBATCH --mem=500000

module purge
module load Python/3.11.3-GCCcore-12.3.0

# Create and load virtual environment
python3 -m venv $HOME/venvs/master_thesis
source $HOME/venvs/master_thesis/bin/activate

# Install dependencies
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# Move cached datasets to the /scratch directory
#export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"

# Move downloaded models and tokenizers to the /scratch directory
export HF_HOME="/scratch/$USER/.cache/huggingface/hub"

# Move cached datasets and downloaded models and tokenisers to the /scratch directory
#export HF_HOME=/scratch/s4410653/hf_cache

# Set PYTORCH_CUDA_ALLOC_CONF environment variable
export "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# The synthetic data source (Control or SBCSAE) is a command line argument
data_source=$1

# Completion models
## No prefix, ChrF
### flan-t5-xl
python3 fine_tune_t5.py -tr data/${data_source}/${data_source}_train.json -dev data/${data_source}/${data_source}_dev.json -te data/${data_source}/${data_source}_test.json -out exp/${data_source}/flan-t5-xl_nopx_fine-tune_chrf -hf google/flan-t5-xl -em chrf
### t5-large
#python3 fine_tune_t5.py -tr data/${data_source}/${data_source}_train.json -dev data/${data_source}/${data_source}_dev.json -te data/${data_source}/${data_source}_test.json -out exp/${data_source}/t5-large_nopx_fine-tune_chrf -hf google-t5/t5-large -em chrf

##"Complete this sentence: ", ChrF
### flan-t5-xl
python3 fine_tune_t5.py -tr data/${data_source}/${data_source}_train.json -dev data/${data_source}/${data_source}_dev.json -te data/${data_source}/${data_source}_test.json -out exp/${data_source}/flan-t5-xl_cts_fine-tune_chrf -hf google/flan-t5-xl -em chrf -px "Complete this sentence: "
### t5-large
#python3 fine_tune_t5.py -tr data/${data_source}/${data_source}_train.json -dev data/${data_source}/${data_source}_dev.json -te data/${data_source}/${data_source}_test.json -out exp/${data_source}/t5-large_cts_fine-tune_chrf -hf google-t5/t5-large -em chrf -px "Complete this sentence: "

# Baseline models
## No prefix, ChrF
### flan-t5-base
#python3 fine_tune_t5.py -tr data/${data_source}/${data_source}_train.json -dev data/${data_source}/${data_source}_dev.json -te data/${data_source}/${data_source}_test.json -out exp/${data_source}/flan-t5-base_nopx_fine-tune_chrf -hf google/flan-t5-base -em chrf
### t5-base
python3 fine_tune_t5.py -tr data/${data_source}/${data_source}_train.json -dev data/${data_source}/${data_source}_dev.json -te data/${data_source}/${data_source}_test.json -out exp/${data_source}/t5-base_nopx_fine-tune_chrf -hf google-t5/t5-base -em chrf

## "Complete this sentence: ", ChrF
### flan-t5-base
#python3 fine_tune_t5.py -tr data/${data_source}/${data_source}_train.json -dev data/${data_source}/${data_source}_dev.json -te data/${data_source}/${data_source}_test.json -out exp/${data_source}/flan-t5-base_cts_fine-tune_chrf -hf google/flan-t5-base -em chrf -px "Complete this sentence: "
### t5-base
python3 fine_tune_t5.py -tr data/${data_source}/${data_source}_train.json -dev data/${data_source}/${data_source}_dev.json -te data/${data_source}/${data_source}_test.json -out exp/${data_source}/t5-base_cts_fine-tune_chrf -hf google-t5/t5-base -em chrf -px "Complete this sentence: "


# Backup scripts
cp fine_tune_t5.py exp/${data_source}/