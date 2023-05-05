import copy

import pandas as pd

from src.preprocess.setup import config
from src.preprocess.feature_filter import do_preprocess
# from src.preprocess.feature_filter import load_data


def generate_label(data: pd.DataFrame, label: config.labelType = None):
    print("generating label")
    data = copy.deepcopy(data)
    if 'label' in data:
        return data
    if label:
        config.set_label(label)
    columns = config.get_label_columns()
    index_map = {}
    index = 0
    labelstr = lambda x: '.'.join(x.to_frame().T[columns].to_numpy()[0])
    # print(labelstr(data.iloc[0, :]))
    for _, row in data.iterrows():
        # print(row)
        # print(row.to_frame().T[columns].to_numpy())
        # print(labelstr(row))
        if labelstr(row) not in index_map:
            index_map[labelstr(row)] = index
            index += 1

    data['label'] = [index_map[labelstr(r)] for i, r in data.iterrows()]
    reverse_map = {}
    for key in index_map:
        reverse_map[index_map[key]] = key
    return data, reverse_map

    # return pd.DataFrame(filtered_df)

def select_top_label(data):
    top_labels = data['label'].value_counts().head(3).index.tolist()
    print(f'top:',top_labels)
    data1 = data[data['label'].isin(top_labels)]
    print(data1)

    # reverse_map = {}
    # for key in index_map:
    #     if index_map[key] in top3_labels:
    #         reverse_map[index_map[key]] = key

    return data1,top_labels





#def generate_label_new(data,label):




# if __name__ == '__main__' :
#     config.set_config(config.speciesType.human, config.chainType.alpha)
#     # data = load_data()
#     generate_label(data, config.labelType.mhc_a)
#     print(data['label'].unique())
