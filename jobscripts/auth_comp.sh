#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition=gpu
#SBATCH --mem=86000

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
## No prefix
### flan-t5-xl
python3 authentic_completion.py -ad data/Aphasia/aphasia_broca_processed.json -out exp/${data_source}/flan-t5-xl_nopx_auth_comp -m google/flan-t5-xl

##"Complete this sentence: "
### flan-t5-xl
python3 authentic_completion.py -ad data/Aphasia/aphasia_broca_processed.json -out exp/${data_source}/flan-t5-xl_cts_auth_comp -m google/flan-t5-xl -px "Complete this sentence: "

# Baseline models
## No prefix
### t5-base
python3 authentic_completion.py -ad data/Aphasia/aphasia_broca_processed.json -out exp/${data_source}/t5-base_nopx_auth_comp -m google-t5/t5-base

## "Complete this sentence: "
### t5-base
python3 authentic_completion.py -ad data/Aphasia/aphasia_broca_processed.json -out exp/${data_source}/t5-base_cts_auth_comp -m google-t5/t5-base -px "Complete this sentence: "


# Backup scripts
cp authentic_completion.py exp/${data_source}/