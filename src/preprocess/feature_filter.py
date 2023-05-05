import copy

import pandas
import pandas as pd
import numpy as np
from src.preprocess.label import generate_label
from src.preprocess.setup import config


#
# df = pd.read_table().rename(
#     columns={'cdr3.alpha': 'cdr3_a_aa', "v.alpha": "v_a_gene", "j.alpha": "j_a_gene",
#              'cdr3.beta': 'cdr3_b_aa', 'v.beta': 'v_b_gene', 'j.beta': 'j_b_gene'})


def load_data(src='../../data/vdjdb_full.tsv', label=False, extra_columns=[], row=None, ):
    """
    fetch data from vdjDB and filter them based pre-defined config.
    config.setConfig must be called before load data.
    """
    print("fetching data stage")
    dtypes = {20: str, 29: str, 30: str}

    df = pd.read_table(filepath_or_buffer=src, dtype=dtypes)
    if row:
        df = df.iloc[:row, :]
    df = df.rename(columns=config.get_columns_mapping())
    df = do_drop_badscore(df)
    filtered_df = df.loc[lambda df: df["species"] == config.get_species_name()] \
        .dropna(subset=config.get_columns()).reindex(copy=True)

    filtered_df = filtered_df.reset_index(drop=True)
    return pd.DataFrame(filtered_df).astype(str)


def select_columns(df, columns=None):
    extra_columns = []
    if config.label is not config.labelType.none:
        if 'label' not in df.columns:
            df, _ = generate_label(df)
        extra_columns.append('label')
    df = df.loc[:, columns if columns is not None else config.get_columns() + extra_columns]
    return df


def generate_pid(df):
    for i in range(len(df['meta.subject.id'])):
        df['meta.subject.id'][i] = df['meta.subject.id'][i] if not pd.isnull(df['meta.subject.id'][i]) else f'PID{i}'
    return df


def compute_similarity(df):
    """
    compute alpha, beta or pair-wise alpha&beta into cumulative_sum to extract feature from features.
    """
    columns = config.get_gene_columns()
    for column in columns:
        df[column] = df[column].map(lambda x: sum([int(c, 36) - int('A', 36) for c in x]))
    return df


def compute_count(df: pd.DataFrame, columns: []):
    if config.label is not config.labelType.none:
        res = df.groupby(columns).agg({'label': 'first'}).reset_index()
        res['count'] = df.groupby(columns).size().reset_index().rename(columns={0: 'count'})['count']
        return res.reset_index(drop=True)
    else:
        return pd.DataFrame(df.groupby(columns).size().reset_index().rename(columns={0: 'count'}))

def tcr_drop(df, limit_len=8):
    return df.drop(df[df[config.get_columns()[0]].map(len) < limit_len].index)


switch_cdr3 = 1


def cdr3_3_split(df):
    if switch_cdr3 == 0:
        return df
    else:
        df1 = df.str[3:-3]
        df1 = df1.drop_duplicates()
        return df1


def giana_preprocess(df: pandas.DataFrame):
    df, to_labels = generate_label(df)
    df = select_columns(df)

    df = compute_count(df, config.get_columns())
    # We need to rename the count as required by Giana
    df = df.rename(columns={'count': 'count..templates.reads.'})
    # print(df)
    df = tcr_drop(df, 10)
    df = df.reset_index(drop=True)
    return df, to_labels


def tcr_preprocess(df):
    df, to_labels = generate_label(df)
    df = compute_count(df, config.get_columns())
    return df, to_labels






def select_fe_preprocess_method(method=None):
    if method is not None:
        config.set_fe_method(method)
    return {
        config.feMethodType.distance_metrics: copy.deepcopy,
        config.feMethodType.giana_features: giana_preprocess,
    }.get(config.fe_method, 'default')


def select_distance_preprocess_method(method=None):
    if method is not None:
        config.set_distance_method(method)
    return {
        config.distanceMethodType.tcrdist: tcr_preprocess,
        config.distanceMethodType.giana: giana_preprocess,
        'default': tcr_preprocess
    }.get(config.distance_method, 'default')


def do_preprocess(data):
    preprocess_method = None
    if config.fe_method == config.feMethodType.distance_metrics:
        preprocess_method = select_distance_preprocess_method()
    else:
        preprocess_method = select_fe_preprocess_method()
    new_data = preprocess_method(data)
    return new_data


def do_drop_badscore(data):
    ret = data[data["vdjdb.score"] != 0]
    return ret


if __name__ == '__main__':
    config.set_config(config.speciesType.human, config.chainType.beta)
    config.set_label(config.labelType.epitope)
    config.set_fe_method(config.feMethodType.distance_metrics)
    config.set_distance_method(config.distanceMethodType.giana)
    data = pandas.DataFrame(load_data().iloc[:200, :])
    data = do_preprocess(data)
    data.to_csv('test.csv')
    print(data)
