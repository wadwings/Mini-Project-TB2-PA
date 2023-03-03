from src.preprocess.setup import config
import pandas as pd
import numpy as np


#
# df = pd.read_table().rename(
#     columns={'cdr3.alpha': 'cdr3_a_aa', "v.alpha": "v_a_gene", "j.alpha": "j_a_gene",
#              'cdr3.beta': 'cdr3_b_aa', 'v.beta': 'v_b_gene', 'j.beta': 'j_b_gene'})


def load_data(src='../../data/vdjdb_full.tsv', fineCut=False, extraColumns=[]):
    """
    fetch data from vdjDB and filter them based pre-defined config.
    config.setConfig must be called before load data.
    """
    print("fetching data stage")
    df = pd.read_table(src).rename(
        columns=config.get_columns_mapping())
    filtered_df = df.loc[lambda df: df["species"] == config.get_species_name()] \
        .dropna(subset=config.get_columns()).reindex(copy=True)
    filtered_df = filtered_df.reset_index()
    if fineCut:
        filtered_df = filtered_df.loc[:, config.get_columns() + extraColumns]
    return pd.DataFrame(filtered_df)


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


def compute_count(df, columns):
    return pd.DataFrame(df.groupby(columns).size().reset_index().rename(columns={0: 'count'}))


def tcr_drop(df, limit_len=8):
    return df.drop(df[df[config.get_columns()[0]].map(len) < limit_len].index)


if __name__ == '__main__':
    config.set_config(config.speciesType.human, config.chainType.pw_ab)
    data = load_data()
    data = compute_similarity(data)
    print(data[config.get_gene_columns()])
