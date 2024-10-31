import argparse
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from transformers import set_seed
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
        "-t",
        "--test_data",
        type=str,
        help="File containing test data",
    )
    parser.add_argument(
        "-p",
        "--prediction_data",
        type=str,
        help="File containing model predictions",
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

    # Define class labels
    logger.info("Defining labels...")
    id2label = {
        0: "Healthy",
        1: "Aphasic",
    }

    # Load test data and predictions
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test data file '{args.test_data}' not found.")
    if not os.path.exists(args.prediction_data):
        raise FileNotFoundError(f"Prediction data file '{args.prediction_data}' not found.")

    logger.info("Loading test data and predictions...")
    try:
        X_dev, y_dev = get_data(args.test_data)
        with open(args.prediction_data) as file:
            y_pred = [int(line.strip()) for line in file]
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Evaluate predictions
    logger.info("Evaluating predictions...")
    print(f"Accuracy: {accuracy_score(y_dev, y_pred)}")
    print(
        classification_report(
            y_dev,
            y_pred,
            digits=3,
            target_names=list(id2label.values()),
        )
    )
    print(pd.crosstab(y_dev, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


if __name__ == "__main__":
    main()
