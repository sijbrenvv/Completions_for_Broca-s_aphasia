import argparse
import spacy_stanza
import spacy_udpipe
import stanza
import random
import string
import numpy as np
from datasets import Dataset, load_dataset
from spacy.matcher import Matcher
from spacy.util import filter_spans
from transformers import set_seed
import logging
import os
import warnings
from pattern.text.en import singularize, pluralize

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_spacy_stanza_pipeline() -> spacy_udpipe:
    """
     Download and load a Spacy-Stanza pipeline for English.

    Returns:
        spacy_stanza.StanzaLanguage: Loaded Spacy-Stanza pipeline.
    """
    try:
        spacy_udpipe.download('en')
        model = spacy_udpipe.load('en')
        return model
    except Exception as e:
        stanza.download('en')
        try:
            model = spacy_stanza.load_pipeline('en')
            return model
        except Exception as e:
            model = spacy_stanza.load_pipeline("xx", lang='en')
            return model


def det_sub(det: str):
    """
    Replace a possessive or demonstrative determiner by another random appropriate determiner.
    Args:
         det (str): The determiner to replace.
    Returns:
        Randomly appropriate determiner.
    """
    for _, det_list in dets.items():
        if det.lower() in det_list:
            # Remove the determiner from the list
            y = [i for i in det_list if det != i]
            return random.choice(y)
    return ""

def count_pos(doc: any, length: int) -> bool:
    """
    Function that counts the number of POS tags, subsequently calculates the open-closed classes ratio,
    and returns whether a type of POS should be added or not.

    :param doc: spacy.Doc with the sentence to process.
    :param length: The length of the processed sentence.
    :return: Boolean: True if number of open classes > closed classes.
    """

    nouns = []
    verbs = []
    determiners = []
    prepositions = []
    adjectives = []
    adverbs = []
    interjections = []
    open_close = np.random.gamma(shape=4.99415, scale=1 / 3.558095)
    add = False

    # Count number of respective POS tags
    for tok in doc:
        if tok.pos_ == "NOUN":
            nouns.append(tok.text)
        elif tok.pos_ == "VERB" or tok.dep_ == "cop" or tok.tag_ in ["VBD", "VBN"]:
            verbs.append(tok.text)
        # det:art and det:dem only
        elif tok.dep_ == "det" and ("Dem" in tok.morph.get('PronType') or "Art" in tok.morph.get('PronType')):
            determiners.append(tok.text)
        elif tok.dep_ == "prep":
            prepositions.append(tok.text)
        elif tok.pos_ == "ADJ":
            adjectives.append(tok.text)
        elif tok.pos_ == "ADV":
            adverbs.append(tok.text)
        elif tok.pos_ == "INTJ":
            interjections.append(tok.pos_)

    open_class_num = len(nouns) + len(verbs) + len(adjectives) + len(adverbs)
    closed_class_num = length - open_class_num - len(interjections)

    # No removing only adding
    if closed_class_num != 0:
        if open_close > open_class_num / closed_class_num:
            add = True

    return add


def symbol_check(text: str) -> bool:
    """
    Check if the provided text contains symbols.

    Args:
        text (str): Input text to check.

    Returns:
        boolean: True if text contains symbols, False otherwise.
    """
    # string.punctuation: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    return any(symbol in text for symbol in string.punctuation.replace(',', '').replace('.', ''))


def nb_np_vp(sen: any) -> tuple[int, int]:
    """ This function returns the number of noun and verb phrases in a SpaCy Doc.

    Args:
        sen (spacy.Doc): Input SpaCy document.

    Returns:
        Tuple: Number of noun phrases and verb phrases.
    """
    nb_np = len([chunk for chunk in sen.noun_chunks])

    # Create a matcher to get the verb phrases
    # 'As of spaCy v3.0, Matcher.add takes a list of patterns as the second argument \
    # instead of a variable number of arguments.'
    patterns = [{'POS': 'VERB', 'OP': '?'},
               {'POS': 'ADV', 'OP': '*'},
               {'POS': 'AUX', 'OP': '*'},
               {'POS': 'VERB', 'OP': '+'}]
    # Instantiate the Matcher instance
    matcher = Matcher(nlp.vocab)
    # Add pattern to matcher
    matcher.add("verb-phrases", [patterns])
    # Call the matcher to find matches
    matches = matcher(sen)
    spans = [sen[start:end] for _, start, end in matches]
    nb_vp = len(filter_spans(spans))

    return nb_np, nb_vp


def keep_sentences(example: any, **fn_kwargs: dict[str:any]) -> dict[str:str]:
    """
    Extract sentences from the example text and filter based on length, symbols, and noun-verb phrase ratio.

    Args:
        example (DatasetDict): Huggingface Dataset object containing the example.
        fn_kwargs: Additional keyword arguments, including 'udpipe' for the language model.
        Data type for 'fn_kwargs' is not mentioned, because it is not compatible with the code.

    Returns:
        string: String containing the sentence to keep.
    """

    # Parse the text
    doc = fn_kwargs["udpipe"](example["preprocessed_text"])

    for sen in doc.sents:
        sen_text = sen.text
        symbol = symbol_check(sen_text)
        sen_len = len(sen_text.split())
        # Continue when the sentence is shorter than or equal to 15 words and does not contain symbols
        if sen_len <= 15 and symbol is False:
            # Extract the number of noun and verb phrases
            nb_np, nb_vp = nb_np_vp(sen)
            if nb_vp == 0:
                continue
            elif nb_np / nb_vp > 2:
                if random.random() > .80:
                    # Add sentence
                    fn_kwargs["ks_dict"]["text"] = sen_text
            else:
                # Add sentence
                fn_kwargs["ks_dict"]["text"] = sen_text
                # To make sure all sentences are stored together in a list for each example
                # use: fn_kwargs["ks_dict"]["text"].append(sen_text)

    return fn_kwargs["ks_dict"]


def make_synthetic(example: any, **fn_kwargs: dict[str:any]) -> dict[str:str]:
    """
     Generate a synthetic sentence based on the input example.

    Args:
        example (DatasetDict): Huggingface Dataset object containing the example.
        fn_kwargs: Additional keyword arguments, including 'udpipe' for the language model.
        Data type for 'fn_kwargs' is not mentioned, because it is not compatible with the code.

    Returns:
        dict: Dictionary containing the synthetic sentences.
    """

    # Parse the text
    doc = fn_kwargs["udpipe"](example["text"])

    # Initialise an empty string to store the synthetic sentence
    temp_syn_sent = ""

    # Uncomment the following if you want to repeat interjection when open classes > closed classes.
    #sen_len = len(example["text"].split())
    #add = count_pos(doc=doc, length=sen_len)

    for token in doc:
        pos = token.pos_

        # Apply different probabilities for token removal based on part-of-speech tags:
        # Discard determiners, particles, and prepositions with 70%
        if pos in {"DET", "ADP", "PART"} or token.dep_ in {"prep"}:
            if random.random() > .70:
                temp_syn_sent += " " + token.text

        # Discard adjectives and adverbs with 50%
        elif pos in {"ADJ", "ADV"}:
            if random.random() < 0.5:
                temp_syn_sent += " " + token.text

        # Lemmatise verbs with 50%
        elif pos in {"AUX", "VERB"}:
            if random.random() > 0.5:
                temp_syn_sent += " " + token.text
                #temp_syn_sent += " " + token.lemma_
            #else:
                #temp_syn_sent += " " + token.text

        # Change noun to plural or singular with 30%
        elif pos in {"NOUN"}:
            # A plural noun for singular in 30% of the times, vice versa.
            if random.random() >= .70:
                if "Plur" in token.morph.get("Number"):
                    temp_syn_sent += " " + singularize(token.text)
                elif "Sing" in token.morph.get("Number"):
                    temp_syn_sent += " " + pluralize(token.text)
            else:
                temp_syn_sent += " " + token.text

        # Modify possessive and demonstrative pronoun with 40%
        elif pos in {"PRON"}:
            if random.random() >= .60:
                if pos == "DET" or "Dem" in token.morph.get('PronType') or "Yes" in token.morph.get('Poss'):
                    sub = det_sub(token.text)
                    temp_syn_sent += " " + sub
                else:
                    temp_syn_sent += " " + token.text
            else:
                temp_syn_sent += " " + token.text

        # Repeat interjections if there are more open word classes
        #elif pos == "INTJ":
            # Repeat pronouns in 10% of the times
        #    if random.random() >= .90 or add:
        #        temp_syn_sent += " " + token.text
        #    temp_syn_sent += " " + token.text

        elif pos in {"PUNCT"}:
            temp_syn_sent += token.text

        else:
            temp_syn_sent += " " + token.text

    # Store the synthetic sentences in the output dictionary if the length requirement is met
    syn_len = len(temp_syn_sent[1:].split())
    org_len = len(example["preprocessed_text"].split())

    if 1/4 * org_len <= syn_len <= 3/4 * org_len:
        fn_kwargs["syn_dict"]["synthetic"] = temp_syn_sent[1:]  # Ignore the leading space
    else:
        fn_kwargs["syn_dict"]["synthetic"] = ""  # Use empty string as filter token

    return fn_kwargs["syn_dict"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path",
        "-inp",
        help="Path to the input data (normal sentences to make aphasic)",
        required=True,
        type=str,
    )
    parser.add_argument(
       "--output_file_path",
       "-out",
       required=True,
       help="Path where to save the output file (synthetic data set).",
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
    parser.add_argument(
        "--huggingface_dataset",
        "-hd",
        # required=True,
        help="Whether the input data is a HuggingFace identifier.",
        default=False,
        type=bool,
    )

    args = parser.parse_args()

    # Set seed for replication
    set_seed(args.random_seed)

    # Define global variables
    dets = {
        'Art': ['a', 'an', 'the'],
        'Dem': ['this', 'that', 'these', 'those'],
        'Poss': ['my', 'your', 'his', 'her', 'its', 'our', 'their']
    }

    if args.huggingface_dataset:
        logger.info("Loading Hugging Face dataset...")
        identifier = args.input_file_path  # For example, 'allenai/c4'
        dataset = load_dataset(identifier)

        # We are only interested in the text column
        column_names = dataset["train"].column_names

        # Remove all columns except the first one: the 'text' column
        dataset = dataset["train"].remove_columns(column_names[1:])

    if not os.path.exists(args.input_file_path):
        raise FileNotFoundError(f"CSV or JSON input file '{args.input_file_path}' not found.\n Did you forget to set -hd to True when using a HF dataset?")

    # Check if the provided path has an extension
    if os.path.splitext(args.input_file_path)[1] == "":
        raise Exception(f"'{args.input_file_path}' contains no extension. Please provide a JSON or CSV file.")

    if args.input_file_path.endswith(".csv"):
        dataset = Dataset.from_csv(args.input_file_path)
    elif args.input_file_path.endswith(".json"):
        dataset = Dataset.from_json(args.input_file_path)

    # Create dictionary for the sentences to keep
    ks_dict = {"text": ""}
    # Get the udpipe model
    logger.info("Loading the udpipe model...")
    nlp = download_spacy_stanza_pipeline()

    # Get all the sentences we want to keep and process further
    logger.info("Retrieving the sentences we want to keep and process further...")
    keep_dataset = dataset.map(
        keep_sentences,
        #batched=True,
        fn_kwargs={"ks_dict": ks_dict, "udpipe": nlp},
    )

    # Filter dataset: remove unkept sentences
    temp_df = keep_dataset.to_pandas()
    filtered_df = temp_df.drop_duplicates(subset=["text"])
    del temp_df
    updated_dataset = Dataset.from_pandas(filtered_df)

    # Create dictionary for the synthetic aphasic sentence and the original one
    syn_dict = {"synthetic": []}#, "original": []}

    # Create a synthetic aphasic sentence for each kept sentence
    logger.info("Generating synthetic aphasic sentences...")
    syn_dataset = updated_dataset.map(
        make_synthetic,
        fn_kwargs={"syn_dict": syn_dict, "udpipe": nlp},
        remove_columns=["text", "__index_level_0__"],
    )

    logger.info("Post-processing...")
    # Filter synthetic dataset: remove empty strings (sents which do not meet length requirement), \
    # and remove one-word and two-word utterances.
    syn_dataset = syn_dataset.filter(lambda example: example["synthetic"] != "")
    syn_dataset = syn_dataset.filter(lambda example: len(example["synthetic"].split()) >= 3)
    # Capitalise each sentence to prevent unwanted behaviour from the completion models
    syn_dataset = syn_dataset.map(lambda example: {"synthetic": example["synthetic"].capitalize()})
    # Capitalise preprocessed_text and remove the space before the full stop
    syn_dataset = syn_dataset.map(
        lambda example: {
            "preprocessed_text": example["preprocessed_text"].capitalize().rstrip(" .") + "."
        }
    )

    # Export dataset
    logger.info(f"Exporting the synthetic dataset...")
    os.makedirs(args.output_file_path, exist_ok=True)
    syn_dataset.to_csv(os.path.join(args.output_file_path, "syn_data.csv"), index=False, sep=',')
    syn_dataset.to_json(os.path.join(args.output_file_path, "syn_data.json"), orient="records", lines=True)
    logger.info(f"Dataset saved to: {os.path.join(args.output_file_path, 'syn_data.[json|csv]')}")

