# Generating Completions for Broca's Aphasic Sentences Using Large Language Models

## Abstract

Broca's aphasia is a type of aphasia characterized by non-fluent, effortful and agrammatic speech production with relatively good comprehension. 
Since traditional aphasia treatment methods are often time-consuming, labour-intensive, and do not reflect real-world conversations, applying natural language processing based approaches such as Large Language Models (LLMs) could potentially contribute to improving existing treatment approaches. 
To address this issue, we explore the use of sequence-to-sequence LLMs for completing Broca's aphasic sentences. 
We first generate synthetic Broca's aphasic data using a rule-based system designed to mirror the linguistic characteristics of Broca's aphasic speech. 
Using this synthetic data (without authentic aphasic samples), we then fine-tune four pre-trained LLMs on the task of completing agrammatic sentences. 
We evaluate our fine-tuned models on both synthetic and authentic Broca's aphasic data. 
We demonstrate LLMs' capability for reconstructing agrammatic sentences, with the models showing improved performance with longer input utterances. 
Our result highlights the LLMs' potential in advancing communication aids for individuals with Broca's aphasia and possibly other clinical populations. 

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

We created a [helper](https://github.com/sijbrenvv/Completions_for_Broca-s_aphasia/blob/main/jobscripts/helper_preprocessing.sh) for the setup and pre-processing steps:
```bash
jobscripts/helper_preprocessing.sh
```

The helper first executes the data setup files, converting the raw CHA files into workable dataframes, and then runs the pre-processing files over these dataframes.

### Generating synthetic sentences and assessing their quality

Similar to the data setup and pre-processing, we created a [helper](https://github.com/sijbrenvv/Completions_for_Broca-s_aphasia/blob/main/jobscripts/helper_data_quality.sh) for generating synthetic sentences and assessing their quality automatically.
```bash
jobscripts/helper_data_quality.sh
```

The helper generates synthetic sentences using the [SBCSAE corpus](https://www.linguistics.ucsb.edu/research/santa-barbara-corpus) and reproduces the data evaluation as shown in Table 3 in the paper.
See the corresponding bash scripts for more information such as the data paths.

### Fine-tuning the models including generation and analysis

Before we can fine-tune the sentence completions models, we need to create the data splits:
```bash
jobscripts/finetuning_splits.sh 31-10-2024
```

The splits can be found in the [data folder](https://github.com/sijbrenvv/Completions_for_Broca-s_aphasia/tree/main/data/SBCSAE).

Next up we fine-tune the sentence completion models, let them generate completions for the test set, and evaluate their performances using our [fine-tune script](https://github.com/sijbrenvv/Completions_for_Broca-s_aphasia/blob/main/jobscripts/fine_tune.sh):
```bash
jobscripts/fine_tune.sh SBCSAE
```

See `python fine_tune_t5.py --help` for more information about its parameters, and please find the generated completions in the [experiment folder](https://github.com/sijbrenvv/Completions_for_Broca-s_aphasia/tree/main/exp/completion/SBCSAE) for convenience.


To gain more insights into the ChrF and Cosine similarity scores for each model, run the following command:
```bash
jobscripts/analyse_comp.sh
```

The bash scripts provides descriptive statistics about the completions by each model, including standard error, effectively recreating Table 4 in the paper.

### Generating completions for authentic Broca's aphasic sentences

The generated completions for the authentic Broca's aphasic sentences can be reproduced using the [authentic completion script](https://github.com/sijbrenvv/Completions_for_Broca-s_aphasia/blob/main/jobscripts/auth_comp.sh):
```bash
jobscripts/auth_comp.sh
```

See the bash script and `python authentic_completion.py --help` to reuse the code with different input sentences.
Please find the generated completions for the authentic input in the [experiment folder](https://github.com/sijbrenvv/Completions_for_Broca-s_aphasia/tree/main/exp/completion/SBCSAE) as well.
