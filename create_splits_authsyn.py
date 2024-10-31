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
        "--authentic_data",
        "-ad",
        required=True,
        help="Path to the authentic aphasic data file.",
        type=str,
    )
    parser.add_argument(
        "--synthetic_data",
        "-sd",
        required=True,
        help="Path to the synthetic aphasic data file.",
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

    # Read the authentic data and add the label = 1
    if not os.path.exists(args.authentic_data):
        raise FileNotFoundError(f"Authentic aphasic data file '{args.authentic_data}' not found.")
    if not os.path.exists(args.synthetic_data):
        raise FileNotFoundError(f"Synthetic aphasic data file '{args.synthetic_data}' not found.")
    logger.info("Loading and processing authentic data...")
    authentic_df = pd.read_json(args.authentic_data, lines=True)
    authentic_df["label"] = 1

    # If the data is authentic, rename the column name to 'text'
    if "preprocessed_text" in authentic_df.columns:
        authentic_df.rename(columns={"preprocessed_text": "text"}, inplace=True)

    # We only need the text and label columns for the classifier
    authentic_df = authentic_df[["text", "label"]]
    logger.info(f"Authentic dataframe shape: {authentic_df.shape}")

    # Read the synthetic data and add the label = 0
    logger.info("Loading and processing synthetic data...")
    synthetic_df = pd.read_json(args.synthetic_data, lines=True)
    synthetic_df["label"] = 0

    # Rename the column 'synthetic' to 'text'
    if "synthetic" in synthetic_df.columns:
        synthetic_df.rename(columns={"synthetic": "text"}, inplace=True)

    # We only need the text and label columns for the classifier
    synthetic_df = synthetic_df[["text", "label"]]
    logger.info(f"Synthetic dataframe shape: {synthetic_df.shape}")

    # Concatenate and balance the data \
    # Get the smallest df size and use that portion of the other df
    min_length = min(len(authentic_df.index), len(synthetic_df.index))
    data_df = pd.concat([authentic_df.head(min_length), synthetic_df.head(min_length)])

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
