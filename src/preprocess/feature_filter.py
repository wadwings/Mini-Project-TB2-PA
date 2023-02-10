from src.preprocess.setup import config
import pandas as pd

df = pd.read_table('../../data/vdjdb_full.tsv').rename(
    columns={'cdr3.alpha': 'cdr3_a_aa', "v.alpha": "v_a_gene", "j.alpha": "j_a_gene",
             'cdr3.beta': 'cdr3_b_aa', 'v.beta': 'v_b_gene', 'j.beta': 'j_b_gene'})


def load_data():
    print("fetching data stage")
    filtered_df = df.loc[lambda df: df["species"] == config.getSpeciesName()].dropna(subset=config.getColumns())
    return pd.DataFrame(filtered_df)


def compute_count(df, columns):
    return pd.DataFrame(df.groupby(columns).size().reset_index().rename(columns={0: 'count'}))
