import pandas as pd

from src.preprocess.setup import config
from src.preprocess.feature_filter import load_data


def generate_label(data: pd.DataFrame, labelType: config.labelType):
    print("generating label")
    config.set_label(labelType)
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
    return data
    # return pd.DataFrame(filtered_df)


if __name__ == '__main__' :
    config.set_config(config.speciesType.human, config.chainType.alpha)
    data = load_data()
    generate_label(data, config.labelType.mhc_a)
    print(data['label'].unique())
