import argparse
import logging
import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)
import torch
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Set the logging level for the sentence_transformers library to WARNING
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Set PYTORCH_CUDA_ALLOC_CONF environment variable
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def tokenize_function(examples: any, **fn_kwargs: dict[str:any]) -> any:
    """
    Function to swiftly tokenise the text using the provided tokeniser in fn_kwargs.
    Args:
        examples: All the texts that will be tokenised.
        fn_kwargs (dict): A dictionary of arguments containing the tokeniser, max length value and which column to tokenise.
    Returns:
         Input IDs (list of integers): The tokenised examples.
    """
    return fn_kwargs["tokenizer"](
        examples[fn_kwargs["col"]],
        padding='longest',
        max_length=fn_kwargs["max_length"],
        truncation=True,
        return_tensors="pt")

def get_data(auth_path: str, random_seed: int) -> pd.DataFrame:
    """
    Function to read dataframe with columns.
    Args:
        auth_path (str): Path to the file containing the authentic data.
    Returns:
        pd.Series, pd.Series: A series containing the source text and a series containing the target text.
    """
    auth_df = pd.read_json(auth_path, lines=True)
    temp_dataset = Dataset.from_pandas(auth_df)

    # Capitalise preprocessed_text and remove the space before the full stop
    temp_dataset = temp_dataset.map(
        lambda example: {
            "preprocessed_text": example["preprocessed_text"].capitalize().rstrip(" .") + "."
        }
    )
    # Remove one-word and two-word utterances
    temp_dataset = temp_dataset.filter(lambda example: len(example["preprocessed_text"].split()) >= 3)

    auth_df = Dataset.to_pandas(temp_dataset)
    del temp_dataset

    # Rename 'preprocessed_text' to 'Source'
    auth_df.rename(columns={"preprocessed_text": "Source"}, inplace=True)

    # We only need the 'Source' column
    auth_df = auth_df[["Source"]]

    return auth_df


def test(auth_data: pd.DataFrame, best_model_path: str, prefix: str) -> list[str]:
    """
    Function to generate completions for the provided fragmented sentences (authentic Broca's aphasic).

    :param auth_data: The authentic sentences in a pandas dataframe.
    :param best_model_path: The path to the best-performing model on the validation set.
    :param prefix: The prefix to prepend to the sentences.
    :return: The generated completions for the authentic Broca's aphasic sentences.
    """
    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Pandas dataframe to huggingface Dataset
    auth_dataset = Dataset.from_pandas(auth_data)

    # Get tokeniser from saved model and load best model
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)

    # Define BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        quantization_dtype=torch.float16,
    )

    if torch.cuda.is_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path, device_map="auto", quantization_config=quantization_config)  # torch_dtype=torch.float16
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path)

    # Use a GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Use gradient checkpointing during inference
    model.gradient_checkpointing_enable()

    if prefix != "":
        # Tokenise using a simple prefix (the same as Misra and colleagues)
        logger.info(f"Adding prefix to test set...")
        auth_dataset = auth_dataset.map(
            lambda example: {
                "Source": prefix + example["Source"]
            }
        )
    tokens = tokenizer(auth_dataset['Source'], padding=True, return_tensors="pt")  #.to(device)

    # Clear memory
    del auth_dataset, auth_data

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Disable gradient calculation
    #with torch.no_grad():
    output = model.generate(**tokens, max_new_tokens=25)

    # Clear memory
    del tokens

    # Return the generated completions
    return tokenizer.batch_decode(output, skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auth_file_path",
        "-ad",
        required=True,
        help="Path to the authentic data (in JSON format).",
        type=str,
    )
    parser.add_argument(
        "--output_file_path",
        "-out",
        required=True,
        help="Path where to save the output file.",
        type=str,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Name of the model to generate completions. Default: 'google/flan-t5-xl'",
        default="google/flan-t5-xl"
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
        "--prefix",
        "-px",
        type=str,
        help="The prefix to use, include colon followed by a space ': '!. Default: ''",
        default=""
    )

    args = parser.parse_args()

    # Set seed for replication
    set_seed(args.random_seed)

    if not os.path.exists(args.auth_file_path):
        raise FileNotFoundError(f"Authentic data file '{args.auth_file_path}' not found.")

    auth_path = args.auth_file_path  # For example, 'test.json'
    model = args.model  # For example, 'google/flan-t5-small'
    pref = args.prefix  # For example, 'Complete this sentence: '

    # Get the data
    logger.info(f"Loading the data...")
    auth_df = get_data(auth_path, args.random_seed)

    output_df = pd.DataFrame({
        "Source": [],
        "Gen_comp": []
    })

    # Test completion model
    logger.info(f"Generating completions for the authentic data...")
    # Split dataframe into n pieces
    for c, sub_df in enumerate(np.array_split(auth_df, 10), start=1):
        logger.info(f"Processing sub df: {c}")
        # Convert data structure to pandas df
        df = pd.DataFrame(sub_df)
        gen_comp = test(auth_data=df, best_model_path=f"models/{model}/{args.random_seed}/best/", prefix=pref)
        temp_df = pd.DataFrame({
            "Source": df['Source'].to_list(),
            "Gen_comp": gen_comp
        })
        output_df = pd.concat([output_df, temp_df])
        del temp_df, df

    # Export dataframe
    output_dir = args.output_file_path.split("_")[0]
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\nExporting dataframe to '{output_dir + '/' + args.output_file_path.split('/')[-1]}.[json|csv]'...\n")
    output_df.to_csv(output_dir + "/" + args.output_file_path.split("/")[-1] + ".csv", index=False, sep=',')
    output_df.to_json(output_dir + "/" + args.output_file_path.split("/")[-1] + ".json", orient="records", lines=True)
