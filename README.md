# Generating Completions for Fragmented Broca's Aphasic Sentences Using Large Language Models

## Outline

### Description


### Data


### Pre-processing


## Replicating our experiment

The [jobscripts](https://github.com/sijbrenvv/Completions_for_Broca-s_aphasia/tree/main/jobscripts) folder contains all the jobscripts (including two helpers) needed to replicate the experiments on the Hábrók server. 
Note that a virtual environment should be created beforehand.

### Installation

For installing the dependencies, execute the following command:
```bash 
pip install -r requirements.txt 
```
The code targets Python 3.10 and 3.11.

### Data setup and pre-processing

Note that the data setup scripts require CHA files from AphasiaBank and SBCSAE. Therefore, first retrieve those files and store them accordingly -- see [helper_preprocessing.sh](https://github.com/sijbrenvv/Completions_for_Broca-s_aphasia/blob/main/jobscripts/helper_preprocessing.sh).

### Generating synthetic sentences and assessing their quality


### Fine-tuning the models including generation and analysis


### Generating completions for authentic Broca's aphasic sentences

