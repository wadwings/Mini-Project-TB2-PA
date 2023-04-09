import copy

from src.preprocess import *
from src.distance_compute import *
import pandas as pd
import csv
import os


def do_features_extraction(data, method=None):
    features_extraction_method = select_method(method)
    return features_extraction_method(data)

def select_method(method=None):
    if method is not None:
        config.set_fe_method(method)
    return {
        config.feMethodType.giana_features: giana_features_extraction,
        config.feMethodType.brute_force: brute_features_extraction,
        config.feMethodType.distance_metrics: use_distance_as_features,
    }.get(config.get_fe_method(), 'default')


def brute_features_extraction(data):
    columns = config.get_columns()  # todo
    data = copy.deepcopy(data)
    for column in columns:
        characters = []
        # Get all the unique characters from the input string
        all_chars = set("".join(str(x) for x in data[column].tolist()))

        # Add the missing characters to the characters list
        characters = list(set(characters + list(all_chars)))
        print(characters)

        char_map = {}
        for i, char in enumerate(characters):
            char_map[char] = i + 1
        print(char_map)

        # Replace the values in the input column with the sum of the values of each character in the string
        data[column] = data[column].map(lambda x: sum([char_map[c] for c in x]) if isinstance(x, str) else x)
        data = data.dropna()

    return data


def giana_features_extraction(data):
    data.to_csv("./file.tsv", sep='\t', index=False, quoting=csv.QUOTE_NONE)
    print(data)
    giana({
        'File': './file.tsv',
        'Mat': True,
    })
    feature_matrix = pd.read_table('./file--RotationEncodingBL62.txt_EncodingMatrix.txt', header=None)
    os.remove('./file.tsv')
    os.remove('./file--RotationEncodingBL62.txt_EncodingMatrix.txt')
    os.remove('./file--RotationEncodingBL62.txt')
    os.remove('./VgeneScores.txt')
    return feature_matrix.iloc[:, 4:]


def use_distance_as_features(data):
    return do_distance_compute(data, data)


def method_test(method):
    config.set_config(config.speciesType.human, config.chainType.beta)
    config.set_label(config.labelType.epitope)
    config.set_fe_method(method)
    data = load_data().iloc[:200, :]
    data = do_preprocess(data)
    return do_features_extraction(data)


if __name__ == '__main__':
    print(method_test(config.feMethodType.giana_features))
