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
        "--input_file_path",
        "-inp",
        required=True,
        help="Path to the input data file in JSON format.",
        type=str,
    )
    parser.add_argument(
        "--output_file_path",
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

    # Read the input data
    if not os.path.exists(args.input_file_path):
        raise FileNotFoundError(f"Input data file '{args.input_file_path}' not found.")
    logger.info("Loading and processing input data...")
    inp_df = pd.read_json(args.input_file_path, lines=True)
    logger.info(f"Input dataframe shape: {inp_df.shape}")

    # Create the train, val/dev and test splits
    # No shuffle needed as train_test_split will shuffle by default
    temp_df, test_df = train_test_split(
            inp_df,
            test_size=0.2,
            random_state=args.random_seed,
        )

    train_df, val_df = train_test_split(
            temp_df,
            test_size=0.125,
            random_state=args.random_seed,
        )
    # Train 70%, dev/val 10%, test 20%

    # Output splits to the predefined folder
    logger.info("Outputting splits to predefined folders...")
    #os.makedirs(args.output_file_path, exist_ok=True)
    train_df.to_json(os.path.join(args.output_file_path, f"{args.output_file_path.split('/')[-2]}_train.json"), orient="records", lines=True)
    val_df.to_json(os.path.join(args.output_file_path, f"{args.output_file_path.split('/')[-2]}_dev.json"), orient="records", lines=True)
    test_df.to_json(os.path.join(args.output_file_path, f"{args.output_file_path.split('/')[-2]}_test.json"), orient="records", lines=True)
