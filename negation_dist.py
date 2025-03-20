import argparse
import pandas as pd
from transformers import set_seed
import logging
import warnings
import os

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


def check_negation(sent: str) -> int:
    """
    Check if the provided sentence contains negation.
    :param sent: The sentence (string) to check for negation.
    :return: 1 if there is negation, 0 if there is not.
    """

    negation_words = {"not", "no", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor", "can't",
                      "don't", "doesn't", "isn't", "wasn't", "weren't", "won't", "shouldn't", "wouldn't", "couldn't"}

    words = sent.lower().split()

    return 1 if any(word in negation_words for word in words) else 0


def negation_list(gen_comp: list, source_sen: list) -> dict[str:list]:
    """
    Get negation presence for all sentence pairs (predicted completions and source sentences).
    Args:
        gen_comp (list): All the generated completions.
        source_sen (list): All the source sentences.
    Returns:
        Dictionary: The presence of negation for all sentence pairs.
    """

    gen_neg = []
    source_neg = []

    for c, comp in enumerate(gen_comp):
        gen_neg.append(check_negation(comp))
        source_neg.append(check_negation(source_sen[c]))

    return {
        "Generated": gen_neg,
        "Source": source_neg
    }


def negation_dist(neg_dict: dict[str:list, str:list]) -> dict:
    """
    Analyse the negation lists: distribution, number of change, additions, and removals.
    :param neg_dict: Dictionary with the negation lists of the generated completions and source sentences.
    :return: Dictionary with the distributions and change of negation.
    """

    gen_list = neg_dict["Generated"]
    source_list = neg_dict["Source"]

    gen_dist = gen_list.count(1) / len(gen_list)
    source_dist = source_list.count(1) / len(source_list)

    # Loop over the lists and retrieve negation addition and removal
    add_count = 0
    rem_count = 0
    for c, v in enumerate(source_list):
        if v == 0 and gen_list[c] == 1:
            add_count += 1
        if v == 1 and gen_list[c] == 0:
            rem_count += 1

    return {
        "Source_dist": source_dist,
        "Gen_dist": gen_dist,
        "Additions": add_count,
        "Removals": rem_count,
        "Neg_change": add_count + rem_count
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
    #parser.add_argument(
    #    "--output_file_path",
    #    "-out",
    #    required=True,
    #    help="Path where to save the output file.",
    #    type=str,
    #)
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

    logger.info(f"Checking for negations...")
    # Get the negation lists (0: no negation; 1: negation)
    neg_dict = negation_list(gen_comp=data["Gen_comp"], source_sen=data["Source"])
    logger.info(f"Computing distributions...")
    # Analyse the negations lists (distributions; number of change, additions, and removals)
    neg_dist = negation_dist(neg_dict=neg_dict)

    # Print dictionary values
    logger.info(f"Outputting...")
    print(f"Distribution of negation in the generated completions: {neg_dist['Gen_dist']}")
    print(f"Distribution of negation in the source sentences: {neg_dist['Source_dist']}")
    print(f"Number of additions: {neg_dist['Additions']}")
    print(f"Number of removals: {neg_dist['Removals']}")
    print(f"Percentage of change: {(neg_dist['Neg_change'] / len(data['Source'])) * 100}%")
