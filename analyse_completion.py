import argparse
import pandas as pd
import numpy as np
from transformers import set_seed
from datasets import Dataset
import logging
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data(file_path: str) -> any:
    """
    Function to read dataframe with columns.
    Args:
        file_path (str): Path to the file containing the validation/development data.
    Returns:
        Dataset object: The data as a Dataset object.
    """
    data_df = pd.read_json(file_path, lines=True)

    return data_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path",
        "-inp",
        help="Path to the input data (json file). For example: 'exp/completion/SBCSAE/flan-t5-xl/flan-t5-xl_fine-tune_chrf.json'.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--random_seed",
        "-seed",
        #required=True,
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

    logger.info(f"Describing the evaluation metrics for '{input_path.split('/')[-1]}'...")
    logger.info(data.ChrF.describe())
    print(data.ChrF.describe(), "\nStandard Error:", data.ChrF.sem())
    logger.info(data.Cos_sim_t5.describe())
    print(data.Cos_sim_t5.describe(), "\nStandard Error:", data.Cos_sim_t5.sem())
    logger.info(data.BLEU.describe())
    print(data.BLEU.describe(), "\nStandard Error:", data.BLEU.sem())
    logger.info(data.RougeL.describe())
    print(data.RougeL.describe(), "\nStandard Error:", data.RougeL.sem())

