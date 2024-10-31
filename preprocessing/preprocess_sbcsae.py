import pandas as pd
import re
import contractions
from remove_repetitions import remove_all_repetitions
import warnings
import logging
import argparse
import os
from helpers import contains_whitespace, make_sentences_df

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_line(utterance: str, mask_pauses: bool, remove_repetitions: bool, remove_masks: bool) -> str:
    """
    Clean text using regular expressions and custom functions.
    Also create masks for filled and unfilled pauses if set true.

    Args:
        utterance (str): The utterance/line to pre-process.
        mask_pauses (boolean): Remove unfilled pauses with <mask> (if set true).
        remove_repetitions (boolean): Remove repetitions, e.g. stuttering (if set true)
        remove_masks (boolean): Remove all masks (set true for healthy speech).
    Returns:
        The pre-processed utterance/line
    """
    expanded_words = []
    for word in utterance.split():
        # using contractions.fix --> he's --> he is
        expanded_words.append(contractions.fix(word))
    utterance = ' '.join(expanded_words)

    # +... Trailing off pause (speaker forgets about what it is about to say)
    unfilled_pauses = ["(..)", "(...)", "+..."]
    for pause in unfilled_pauses:
        #utterance = utterance.replace(pause, "UNFILLEDPAUSE")
        # Remove unfilled pauses instead
        utterance = utterance.replace(pause, "")

    # Replace filler pauses with FILLERPAUSE
    filler_pauses = ["&-um", "&-uh", "&-er", "&-mm" "&-eh", "&-like", "&-youknow", "&-hm", "&-sighs"]
    for pause in filler_pauses:
        #utterance = utterance.replace(pause, "FILLERPAUSE")
        # Remove all filler pauses instead
        utterance = utterance.replace(pause, "")

    # Remove all actions: (e.g. &=points:picture)
    r = re.findall(r"\W\W\w+\W\w+", utterance)
    for regex in r:
        if not contains_whitespace(regex):
            utterance = utterance.replace(regex, "")

    # Remove all unicode errors   \W\d+\w\d+\W
    r = re.findall(r"\W\d+\w\d+\W", utterance)
    for regex in r:
        if not contains_whitespace(regex):
            utterance = utterance.replace(regex, "")

    # Remove all characters that start with 2 special chars and have text succeeding it:
    r = re.findall(r"\W\W\w+", utterance)
    for regex in r:
        # if regex[0] != " " and regex[1] != " ":
        if not contains_whitespace(regex):
            # This prevents second and first letter being a space to be removed.
            # e.g. : ") Cinderella" removes the word Cinderella too, which we do not want.
            utterance = utterance.replace(regex, "")

    # Remove anything between [ and ]
    r = re.findall(r"\[(.*?)\]", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove anything between < and >
    r = re.findall(r"\<(.*?)\>", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove strings &+
    r = re.findall(r"\&+[a-zA-Z]+", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove strings starting with *
    r = re.findall(r"\*[A-Za-z]+", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove strings starting with ʔ
    r = re.findall(r"\ʔ[A-Za-z]+", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove string of Xs with an arbitrary length
    r = re.findall(r"X+", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove anything starting with U<any char or none> (Remove 'YOU' from output)
    r = re.findall(r"U.*", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove any single uppercase letter:
    r = re.findall(r"[A-z][-⌈⌉⌊⌋] ", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove ⌈ followed by a number
    r = re.findall(r"\⌈[0-9]+", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove ⌉ followed by a number
    r = re.findall(r"\⌉[0-9]+", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove ⌊ followed by a number
    r = re.findall(r"\⌊[0-9]+", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove ⌋ followed by a number
    r = re.findall(r"\⌋[0-9]+", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    special_characters = ['(.)', '[/]', '[//]', '‡', 'xxx', '+< ', '„', '+', '"" /..""', '+"/.', '+"', '+/?', '+//.',
                          '+//?', '[]', '<>', '_', '-', '^', ')', '(', ':', 'www .', '*PAR', '+/', '@o', '<', '>',
                          '//..', '//', '/..', '/', '"', 'ʌ', '..?', '0.', '0 .', '"" /.', '⌈', '⌉', '&{l=@', '&}l=@',
                          '⌊', '⌋', 'Ϋ', '=@']

    # Remove remaining special chars
    for special_character in special_characters:
        utterance = utterance.replace(special_character, "")

    utterance = re.sub(' +', ' ', utterance)

    # Remove special characters from starting sentence
    remove_startswiths = [" ", ",", "!", ".", "?", "."]
    for start_string in remove_startswiths:
        if utterance.startswith(start_string):
            utterance = utterance[1:]

    # Replace all pauses with <mask> (if set true)
    if mask_pauses:
        utterance = utterance.replace("UNFILLEDPAUSE", "<mask>")
        utterance = utterance.replace("FILLERPAUSE", "<mask>")

    # Removes all masks (set true for healthy speech)
    if remove_masks:
        utterance = utterance.replace("<mask>", "")

    # Removes stuttering and bigram stuttering
    if remove_repetitions:
        utterance = remove_all_repetitions(utterance)

    utterance = re.sub(' +', ' ', utterance)  # Remove final whitespaces (e.g. double space)

    return utterance


def preprocess_dataset(input_dataset_filename: str, mask_pauses: bool, remove_repetitions: bool, remove_masks: bool) -> pd.DataFrame:
    """
    Pre-process the aphasic ('*PAR')  data (from AphasiaBank) provided in the CSV file line by line.

    Args:
        input_dataset_filename (str): The path to the CSV file with the data to be pre-processed.
        mask_pauses (boolean): Remove unfilled pauses with <mask> (if set true).
        remove_repetitions (boolean): Remove repetitions, e.g. stuttering (if set true)
        remove_masks (boolean): Remove all masks (set true for healthy speech).
    Returns:
         df (DataFrame Object): Dataframe with the pre-processed data.
    """
    df = pd.read_csv(input_dataset_filename, encoding='utf8')
    df = df.dropna()

    # We use all the speakers and all scenario's

    preprocessed_text = []
    for text in df['text']:
        preprocessed_line = preprocess_line(text, mask_pauses, remove_repetitions, remove_masks)
        preprocessed_text.append(preprocessed_line)

    df['original_text'] = df['text']
    df['preprocessed_text'] = preprocessed_text
    df = df.dropna(subset=['preprocessed_text'])

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-inp",
        required=True,
        help="Path to the .csv file to be pre-processed.",
        type=str,
    )
    parser.add_argument(
        "--output_file_path",
        "-out",
        required=True,
        help="Path where to save the pre-processed file.",
        type=str,
    )
    parser.add_argument(
        "--mask_pauses",
        "-mp",
        #required=True,
        help="Whether to mask pauses or not. Default: False",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--remove_repetitions",
        "-rr",
        #required=True,
        help="Whether to remove repetitions or not. Default: False",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--remove_masks",
        "-rm",
        #required=True,
        help="Whether to remove masks or not. Default: False",
        default=False,
        type=bool,
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"CSV file to be pre-processed '{args.input_file}' not found.")

    # Check if the provided path has the CSV extension
    if not os.path.splitext(args.input_file)[1] == ".csv":
        raise Exception(f"'{args.input_file}' is not in CSV format (extension). Please provide a CSV file.")

    logger.info(f"Pre-processing {args.input_file}...")
    preprocessed_df = preprocess_dataset(args.input_file, args.mask_pauses, args.remove_repetitions, args.remove_masks)
    output_df = make_sentences_df(preprocessed_df)
    try:
        output_df = preprocessed_df.drop(columns=['line_information', 'speaker_status', 'line_number', 'utterance_count', 'text'])
    except KeyError:
        output_df = preprocessed_df.drop(columns=['line_information', 'line_number', 'utterance_count', 'text'])
    if args.output_file_path.endswith(".csv"):
        output_df.to_csv(args.output_file_path)
        output_df.to_json(args.output_file_path[:-4] + ".json", orient="records", lines=True)
    else:
        output_df.to_csv(args.output_file_path + ".csv")
        output_df.to_json(args.output_file_path + ".json", orient="records", lines=True)
