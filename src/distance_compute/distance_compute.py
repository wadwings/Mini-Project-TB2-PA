import pandas as pd

from src.preprocess.setup import config
from src.preprocess.feature_filter import load_data, compute_count
from tcrdist.repertoire import TCRrep


def compute_distance(df1, df2=None):
    # print("compute distance stage")
    tr = TCRrep(cell_df=df1,
                organism=config.getSpecies(),
                chains=config.getChain(),
                db_file='alphabeta_gammadelta_db.tsv',
                compute_distances=False)
    if df2 is None:
        tr.compute_distances()
        return tr

    tr2 = TCRrep(cell_df=df2,
                 organism=config.getSpecies(),
                 chains=config.getChain(),
                 db_file='alphabeta_gammadelta_db.tsv',
                 compute_distances=False)
    tr.compute_rect_distances(df=tr.clone_df, df2=tr2.clone_df)
    return tr


def compute_single_distance(df1: pd.Series, df2: pd.Series):
    df1 = df1.to_frame().T
    df2 = df2.to_frame().T
    tr = compute_distance(df1, df2)
    return tr.rw_alpha[0][0]


def example():
    config.setConfig(config.speciesType.human, config.chainType.alpha)
    data = load_data()
    data = compute_count(data, config.getColumns())
    print(data.iloc[:200, :])
    tr = compute_distance(df1=data.iloc[:200, :], df2=data)
    print(tr.rw_alpha)


if __name__ == "__main__":
    example()
