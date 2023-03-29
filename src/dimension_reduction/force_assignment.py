from src.preprocess.setup import config
from src.preprocess.feature_filter import load_data
from src.preprocess.label import generate_label
def replace_string_with_sum(data, column):

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




config.set_config(config.speciesType.human, config.chainType.alpha)
data = load_data('../../data/vdjdb_full.tsv')
print(data)
generate_label(data, config.labelType.mhc_class)

data = data[['cdr3_a_aa', 'v_a_gene', 'j_a_gene', 'label']]
print(data)
data = replace_string_with_sum(data, 'cdr3_a_aa')
data = replace_string_with_sum(data, 'v_a_gene')
data = replace_string_with_sum(data, 'j_a_gene')
print(data)

ncols = data.shape[1]
feature = data.values[:,0:ncols-1]
target = data.values[:,ncols-1]