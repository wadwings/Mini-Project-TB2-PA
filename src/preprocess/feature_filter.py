from src.preprocess.setup import config
import pandas as pd
import numpy as np
#
# df = pd.read_table().rename(
#     columns={'cdr3.alpha': 'cdr3_a_aa', "v.alpha": "v_a_gene", "j.alpha": "j_a_gene",
#              'cdr3.beta': 'cdr3_b_aa', 'v.beta': 'v_b_gene', 'j.beta': 'j_b_gene'})


def load_data(src='../../data/vdjdb_full.tsv'):
    """
    fetch data from vdjDB and filter them based pre-defined config.
    config.setConfig must be called before load data.
    """
    print("fetching data stage")
    df = pd.read_table(src).rename(
        columns=config.get_columns_mapping())
    filtered_df = df.loc[lambda df: df["species"] == config.get_species_name()] \
        .dropna(subset=config.get_columns()).reindex(copy=True)
    return pd.DataFrame(filtered_df)


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


if __name__ == '__main__':
    config.set_config(config.speciesType.human, config.chainType.pw_ab)
    data = load_data()
    data = compute_similarity(data)
    print(data[config.get_gene_columns()])
