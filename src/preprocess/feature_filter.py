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


def gliph_p_file(file_path='./file.tsv', save_path='./gliph_p_file.tsv'):
    df = pd.read_table(file_path).loc[:, ['CDR3b', 'TRBV', 'TRBJ','CDR3a','TRAV','TRAJ','meta.subject.id']]
    df['TRBV'] = df['TRBV'].str.replace(r'\*01', '', regex=True)
    df = df[~df['TRBV'].str.contains('\*')]
    df['TRBJ'] = df['TRBJ'].str.replace(r'\*01', '', regex=True)
    df['TRAV'] = df['TRAV'].str.replace(r'\*01', '', regex=True)
    df['TRAJ'] = df['TRAJ'].str.replace(r'\*01', '', regex=True)
    df = df.rename(columns={'meta.subject.id': 'Patient'})
    df['Patient'] = df['Patient'].str.replace(' ', '')
    df_count = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'Counts'})
    df = df.drop_duplicates().merge(df_count, on=df.columns.tolist(), how='left')
    df['Counts'] = df['Counts'].fillna(1)
    # df = df.fillna('')
    df['Counts'] = df['Counts'].astype(int)

    df.to_csv(save_path, sep='\t', index=False)


if __name__ == '__main__':
    config.set_config(config.speciesType.human, config.chainType.pw_ab)
    data = load_data()
    data = compute_similarity(data)
    print(data[config.get_gene_columns()])
    gliph_p_file('./file.tsv')

