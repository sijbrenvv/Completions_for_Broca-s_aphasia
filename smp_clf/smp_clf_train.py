import argparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
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
    """
    Main function to train and save the classifier model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model to use. See choices for the choices. Default: 'NB'",
        choices=["SVC", "SVM", "DT", "RF", "NB"],
        default="NB",
    )
    parser.add_argument(
        "-tr",
        "--train_file",
        type=str,
        help="File containing training data",
    )
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        help="Location where trained model is saved",
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

    # Get the train data
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Train file '{args.train_file}' not found.")
    logger.info("Loading the train data...")
    X_train, y_train = get_data(args.train_file)

    classifiers = {
        "SVC": LinearSVC(),
        "SVM": SVC(kernel="rbf"),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(n_jobs=-1, random_state=args.random_seed),
        "NB": MultinomialNB(),
    }

    # Get the classic classifier to use
    logger.info("Loading model...")
    model = classifiers[args.model]

    # Initialise the vectoriser using Tf-Idf and unigrams
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 1),
        max_df=0.999,
        min_df=0.001,
    )

    # Transform the text to binary code
    logger.info("Encoding the train data...")
    X_train = vectorizer.fit_transform(X_train)

    # Train the model
    logger.info("Training model...")
    model.fit(X_train, y_train)

    # Save the model
    logger.info("Saving model...")
    output_path = os.path.join(args.output_file, "model.pkl")
    os.makedirs(args.output_file, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Model saved to: {output_path}")

    # Save the vectorizer
    logger.info("Saving vectoriser...")
    output_path = os.path.join(args.output_file, "vectorizer.pkl")
    joblib.dump(vectorizer, output_path)
    logger.info(f"Vectoriser saved to: {output_path}")


if __name__ == "__main__":
    main()

