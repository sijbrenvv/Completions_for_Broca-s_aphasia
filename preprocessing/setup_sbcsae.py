import pandas as pd
import os
import re
import logging
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cha_to_df(file_name: str) -> pd.DataFrame:
    """
    Return a dataframe that converts single .cha (.txt) file to a dataframe.

    Args:
        file_name (string): .txt file containing (chat formatted).
    Returns:
        df (DataFrame object): Pandas dataframe object of the CHAT formatted TXT file
    """
    f = open(file_name, "r", encoding="utf-8")
    lines = f.readlines()

    scenarios = []
    current_scenario = "N/A"

    # Loop over '@Participants:' to extract the speakers: check if they are a speaker and add the tag to line_type (or change variable name)
    # Ignore 'X' as speaker
    line_type = []
    participants = lines[3].rstrip().split("\t")[1].split(",")
    for ent in participants:
        ents = ent.split(" ")
        # Remove preceding space / empty string
        ents = [i for i in ents if i != ""]
        if ents[2] == "Speaker" and ents[0] != "X":
            line_type.append(f"*{ents[0]}")

    #current_line_type = "N/A"

    line_number = []
    text = []

    i = 0
    for c, line in enumerate(lines):

        # No scenario's involved in the files. Yet each file can be seen as a scenario.
        # Therefore, add first instance of '@Comment:' as the scenario
        if line[:9] == "@Comment:" and lines[c-1][:9] != "@Comment:":
            line = re.sub("\n", "", line)
            line = re.sub("\t", " ", line)
            current_scenario = line[9:].strip()
            if current_scenario[-1] == ".":
                current_scenario = current_scenario[:-1]
        scenarios.append(current_scenario)

        # Add checker for the length of the tag and modify slices accordingly
        col_i = line.find(":")
        idx = col_i if col_i != -1 and col_i < 5 else 5

        # if line[:4] in line_type:
        if line[:idx] in line_type:
            #if line[:idx] != current_line_type:
            #current_line_type = line[:idx]
            i += 1
        line_number.append(i)

        # Modify code: put each utterance on a line (split on: newline?, ".", "?" and "!"?)
        # Make sure "line_number" and "text" align, i.e. add line-number for every text added.
        line = re.sub("\n", "", line)
        line = re.sub("\t", " ", line)
        text.append(line)

    columns = ['line_number', 'scenario', 'text', 'line_information', 'utterance_count']
    df = pd.DataFrame(columns=columns)
    df['line_number'], df['scenario'], df['text'] = line_number, scenarios, text
    df = df.groupby(['line_number', 'scenario'])['text'].apply(' '.join).reset_index()

    #df['line_information'] = df['text'].astype(str).str[:4]
    #df['text'] = df['text'].str[6:]
    df['line_information'] = df['text'].astype(str).str[:5]
    df['text'] = df['text'].str[5:]
    df = df.loc[df['line_information'] != "@Begi"]
    df = df.loc[df['line_information'] != "@G:  "]
    df = df.loc[df['line_information'] != "@UTF8"]
    df = df.loc[df['line_information'] != "@Comm"]
    df = df.loc[df['line_information'] != "@End\n"]

    utterance_number = []
    utterance_count = 0

    for info in df['line_information']:
        # Participants here is the same as line_type defined at the beginning
        #participants = ["*INV", "*PAR", "*IN1", "*IN2"]
        participants = line_type
        if info in participants:
            utterance_count += 1

        utterance_number.append(utterance_count)

    df['utterance_count'] = utterance_number

    return df

# Perhaps import 'cha_to_csv' from 'setup_ab.py' if it is not modified at all!
def cha_to_csv(data_dir: str, file_name: str) -> bool:
    """
    Loop over a directory containing .cha based .txt files to convert to a .csv file.

    Args:
        data_dir (str): Directory in which the .cha files are located.
        file_name (str): Path to the .csv file to save the data in.
    Returns:
         Boolean: true if completed.
    """
    columns = ['line_number', 'scenario', 'text', 'line_information', 'utterance_count', 'source_file']
    df = pd.DataFrame(columns=columns)

    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            file_df = cha_to_df(data_dir + file)
            file_df['source_file'] = file
            df = pd.concat([df, file_df]) #df.append(file_df)
    df.to_csv(file_name, index=False, encoding="utf-8")
    logger.info(f"CSV file saved to: {str(file_name)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "-data",
        required=True,
        help="Path to the data directory containing the .cha files to be converted into one .csv file.",
        type=str,
    )
    parser.add_argument(
        "--output_file_path",
        "-out",
        required=True,
        help="Path where to save the output file (csv file).",
        type=str,
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory '{args.data_dir}' not found.")

    #data_dir = "data/Control/all_files/"
    #csv_filename = "data/Control/control_broca.csv"
    logger.info(f"Converting all .cha files in '{args.data_dir}' to a single .csv at '{args.output_file_path}'")
    cha_to_csv(args.data_dir, args.output_file_path)
