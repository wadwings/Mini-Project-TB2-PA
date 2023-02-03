from setup import config
import pandas as pd

df = pd.read_table('../../data/vdjdb_full.tsv').rename(
    columns={'cdr3.alpha': 'cdr3_a_aa', "v.alpha": "v_a_gene", "j.alpha": "j_a_gene",
             'cdr3.beta': 'cdr3_b_aa', 'v.beta': 'v_b_gene', 'j.beta': 'j_b_gene'})

def filtered_data():
    print("fetching data stage")
    filtered_df = df.loc[lambda df: df["species"] == config.getSpeciesName()].dropna(subset=config.getColumns()).loc[:,
                  config.getColumns()]
    filtered_df = filtered_df.groupby(filtered_df.columns.tolist()).size().reset_index().rename(columns={0: 'count'})
    return filtered_df
