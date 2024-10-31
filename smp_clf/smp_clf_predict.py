import argparse
import pandas as pd
from transformers import set_seed
import joblib
import logging
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data(file_path: str) -> tuple[pd.Series, pd.Series]:
    """
    Function to read dataframe with columns.
    Args:
        file_path (str): Path to the file containing the data.
    Returns:
        pd.Series, pd.Series: A series containing the text and a series containing the label columns.
    """
    file_df = pd.read_json(file_path, lines=True)
    return file_df["text"], file_df["label"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Name of file containing a trained model",
    )
    parser.add_argument(
        "-inp",
        "--input_file",
        type=str,
        help="Input file containing data",
    )
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        help="Name of output file",
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

    # Get the dev data
    logger.info("Loading input data...")
    X_dev, _y_dev = get_data(args.input_file)

    # Load the trained model
    logger.info("Loading model...")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file '{args.model}' not found.")
    model = joblib.load(args.model)

    # Load the corresponding vectoriser
    logger.info("Loading corresponding vectoriser...")
    vectorizer = joblib.load(args.model.replace("model", "vectorizer"))

    # Use the loaded model to predict on dev data
    logger.info("Making predictions...")
    X_dev = vectorizer.transform(X_dev)
    y_pred = model.predict(X_dev)

    # Save predictions to predefined output file
    logger.info("Saving predictions to output file...")
    with open(args.output_file, "w") as file:
        file.write("\n".join(map(str, y_pred)))


if __name__ == "__main__":
    main()

