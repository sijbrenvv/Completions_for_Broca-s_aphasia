""" Helper functions during the pre-processing of the spoken language"""
import pandas as pd


def contains_whitespace(string: str) -> bool:
    """
    Check if string contains whitespace.

    Args:
        string: input string.
    Returns:
         boolean: True if string contains a space, else False.
    """
    for letter in string:
        if letter == " ":
            return True
    return False


def return_ending_punctuation(input_sentence: str) -> str:
    """
    Return ending punctuation '?', '!' or '.' for a provided sentence.

    Args:
        input_sentence: string.
    Returns:
         symbol (str): string of the ending punctuation.
    """
    matches = ["?", "!", "."]

    for symbol in input_sentence[-3:]:
        if symbol in matches:
            return str(symbol)


def make_sentences_df(input_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Converts incoherent lines into sentences based on sentence ending, e.g: '?', '.' or '!'
    (Incoherent lines are lines that should belong together, but have been split previously)

    Args:
        input_dataset (DataFrame object): Data set containing the lines to sentencise.
    Returns:
         DataFrame Object: dataframe with the sentences.
    """
    df = input_dataset
    df = df.dropna(subset=['preprocessed_text'])
    try:
        df = df.drop(columns=['line_information', 'speaker_status', 'line_number', 'utterance_count'])
    except KeyError:
        df = df.drop(columns=['line_information', 'line_number', 'utterance_count'])
    line_merge_number = []
    i = 0
    for sentence in df['preprocessed_text']:
        matches = ["?", "!", "."]
        if any([x in sentence for x in matches]):
            line_merge_number.append(i)
            i += 1
        else:
            line_merge_number.append(i)

    df['line_merge_number'] = line_merge_number

    df2 = pd.DataFrame(columns=df.columns)
    for number in df['line_merge_number'].unique():
        tobe_merged_data = df.loc[(df['line_merge_number'] == number)]
        if len(tobe_merged_data) <= 1:
            df2 = pd.concat([df2, tobe_merged_data])   # df2.append(tobe_merged_data)
        else:
            original_text_merged = ''.join(tobe_merged_data['text'].to_list())
            processed_text_merged = ''.join(tobe_merged_data['preprocessed_text'].to_list())
            if processed_text_merged[:-1] == 0:
                new_row = {'scenario': tobe_merged_data['scenario'].values[0],
                           'text': original_text_merged,
                           'source_file': tobe_merged_data['source_file'].values[0],
                           'preprocessed_text': processed_text_merged,
                           'line_merge_number': None
                           }
                df2 = pd.concat([df2, new_row], ignore_index=True)  # df2.append(new_row, ignore_index=True)

    df2 = df2.drop(columns=['text', 'line_merge_number'])
    return df2
