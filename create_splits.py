import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import set_seed
import logging
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aphasic_data",
        "-ad",
        required=True,
        help="Path to the aphasic data file.",
        type=str,
    )
    parser.add_argument(
        "--healthy_data",
        "-hd",
        required=True,
        help="Path to the healthy controls data file.",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        "-out",
        required=True,
        help="Path where to save the data splits.",
        type=str,
    )
    parser.add_argument(
        "--random_seed",
        "-seed",
        #required=True,
        help="The random seed to use.",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    # Set seed for replication
    set_seed(args.random_seed)

    # Read the aphasic data and add the label = 1
    if not os.path.exists(args.aphasic_data):
        raise FileNotFoundError(f"Aphasic data file '{args.aphasic_data}' not found.")
    logger.info("Loading and processing aphasic data...")
    aphasic_df = pd.read_json(args.aphasic_data, lines=True)
    aphasic_df["label"] = 1
    # If the data is synthetic, rename the column name to 'text'
    if "synthetic" in aphasic_df.columns:
        aphasic_df.rename(columns={"synthetic": "text"}, inplace=True)

    # If the data is authentic, rename the column name to 'text'
    elif "preprocessed_text" in aphasic_df.columns:
        aphasic_df.rename(columns={"preprocessed_text": "text"}, inplace=True)

    # We only need the text and label columns for the classifier
    aphasic_df = aphasic_df[["text", "label"]]
    logger.info(f"Aphasic dataframe shape: {aphasic_df.shape}")

    # Read the healthy data and add the label = 0
    if not os.path.exists(args.healthy_data):
        raise FileNotFoundError(f"Control data file '{args.healthy_data}' not found.")
    logger.info("Loading and processing control data...")
    healthy_df = pd.read_json(args.healthy_data, lines=True)
    healthy_df["label"] = 0

    # The healthy data is always authentic
    healthy_df.rename(columns={"preprocessed_text": "text"}, inplace=True)

    # We only need the text and label columns for the classifier
    healthy_df = healthy_df[["text", "label"]]
    logger.info(f"Healthy dataframe shape: {healthy_df.shape}")

    # Concatenate and balance the data \
    # The healthy dataframe is much larger than the aphasic one
    # Get the smallest df size and use that potion of the healthy df
    min_length = min(len(aphasic_df.index), len(healthy_df.index))
    data_df = pd.concat([aphasic_df.head(min_length), healthy_df.head(min_length)])

    # No shuffle needed as train_test_split will shuffle by default
    train_df, val_df = train_test_split(
            data_df,
            test_size=0.2,
            random_state=args.random_seed,
        )

    # Output splits to the predefined folder
    logger.info("Outputting splits to predefined folders...")
    os.makedirs(args.output_file, exist_ok=True)
    train_df.to_json(os.path.join(args.output_file, "train.json"), orient="records", lines=True)
    logger.info(f"Train split saved to: {os.path.join(args.output_file, 'train.json')}")
    val_df.to_json(os.path.join(args.output_file, "dev.json"), orient="records", lines=True)
    logger.info(f"Validation split saved to: {os.path.join(args.output_file, 'dev.json')}")
