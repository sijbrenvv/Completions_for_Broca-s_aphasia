import argparse
import pandas as pd
import numpy as np
from transformers import set_seed
from datasets import Dataset
import logging
import warnings
import os
import evaluate

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data(file_path: str) -> pd.DataFrame:
    """
    Function to read dataframe with columns.
    Args:
        file_path (str): Path to the file containing predicted and target completions.
    Returns:
        Pandas DataFrame: The data as a pd DataFrame object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV or JSON input file '{file_path}' not found")

    # Check if the provided input path has a valid extension
    if os.path.splitext(file_path)[1] not in {".json", ".jsonl", ".csv"}:
        raise Exception(f"'{file_path}' contains no valid extension. Please provide a JSON(L) or CSV file.")

    if file_path.endswith(".csv"):
        return pd.DataFrame.from_csv(file_path)
    elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
        return pd.read_json(file_path, lines=True)


def evaluate_comp(gen_comp: list, tar_comp: list) -> dict[str:list]:
    """
    Evaluate the predicted completions against the target completions.
    Args:
        gen_comp (list): All the generated completions.
        tar_comp (list): All the target completions.
    Returns:
        Dictionary: The BLEU and RougeL scores for each sentence pair.
    """
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    bleu_scores = []
    rouge_scores = []
    for c, v in enumerate(gen_comp):
        bleu_scores.append(bleu.compute(predictions=[v], references=[tar_comp[c]])['bleu'])
        rouge_scores.append(rouge.compute(predictions=[v], references=[tar_comp[c]])['rougeL'])

    return {
        "BLEU": bleu_scores,
        "RougeL": rouge_scores
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path",
        "-inp",
        help="Path to the input data (json.csv file). For example: 'exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_fine-tune_chrf.json'.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--random_seed",
        "-seed",
        help="The random seed to use. Default: 0",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    # Set seed for replication
    set_seed(args.random_seed)

    if not os.path.exists(args.input_file_path):
        raise FileNotFoundError(f"Input file '{args.input_file_path}' not found.")

    input_path = args.input_file_path

    # Get the data for the analyses
    logger.info(f"Loading the data...")
    data = get_data(input_path)

    eval_sc = evaluate_comp(gen_comp=data["Gen_comp"], tar_comp=data["Target"])
    data["BLEU"] = eval_sc["BLEU"]
    data["RougeL"] = eval_sc["RougeL"]

    # Export dataframe
    output_dir = args.output_file_path.split("_")[0]
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\nExporting dataframe to '{output_dir + '/' + args.output_file_path.split('/')[-1]}.[json|csv]'...\n")
    data.to_csv(output_dir + "/" + args.output_file_path.split("/")[-1] + ".csv", index=False, sep=',')
    data.to_json(output_dir + "/" + args.output_file_path.split("/")[-1] + ".json", orient="records", lines=True)


