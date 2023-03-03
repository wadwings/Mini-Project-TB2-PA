import os

import pandas as pd
import numpy as np
import csv

from src.preprocess.setup import config
from src.preprocess.feature_filter import load_data, compute_count, tcr_drop
from tcrdist.repertoire import TCRrep
from src.distance_compute.giana.GIANA4 import giana
from numpy.linalg import norm

def tcrdist_method(df1, df2):
    tr = TCRrep(cell_df=df1,
                organism=config.get_species(),
                chains=config.get_chain(),
                db_file='alphabeta_gammadelta_db.tsv',
                compute_distances=False)
    if df2 is None:
        tr.compute_distances()
        return tr

    tr2 = TCRrep(cell_df=df2,
                 organism=config.get_species(),
                 chains=config.get_chain(),
                 db_file='alphabeta_gammadelta_db.tsv',
                 compute_distances=False)
    tr.compute_rect_distances(df=tr.clone_df, df2=tr2.clone_df)
    if config.chain == config.chainType.alpha:
        return tr.rw_alpha
    if config.chain == config.chainType.beta:
        return tr.rw_beta
    if config.chain == config.chainType.pw_ab:
        return tr.rw_alpha, tr.rw_beta,


def gliph_method(df1, df2=None):
    return None


def giana_method(df1, df2):
    df1.to_csv("./file.tsv", sep='\t', index=False, quoting=csv.QUOTE_NONE)
    giana({
        'File': './file.tsv',
        'Mat': True,
    })
    feature_matrix = pd.read_table('./file--RotationEncodingBL62.txt_EncodingMatrix.txt', header=None)
    tcr_feature = {}
    for index, row in feature_matrix.iterrows():
        tcr_feature[row[0]] = row[3:]
    distance_matrix = []
    for x, x_row in df1.iterrows():
        distance_matrix.append([])
        for y, y_row in df1.iterrows():
            distance_matrix[x].append(norm(tcr_feature[x_row[0]] - tcr_feature[y_row[0]]))

    distance_matrix = np.array(distance_matrix)
    os.remove('./file.tsv')
    os.remove('./file--RotationEncodingBL62.txt_EncodingMatrix.txt')
    os.remove('./file--RotationEncodingBL62.txt')
    os.remove('./VgeneScores.txt')
    return distance_matrix

def method_selection(case):
    return {
        config.methodType.tcrdist: tcrdist_method,
        config.methodType.giana: giana_method,
        config.methodType.gliph: gliph_method,
        'default': tcrdist_method,
    }.get(case, 'default')


def compute_distance(df1, df2=None):
    print("compute distance stage")
    return method_selection(config.method)(df1, df2)

def compute_single_distance(df1: pd.Series, df2: pd.Series):
    df1 = df1.to_frame().T
    df2 = df2.to_frame().T
    tr = compute_distance(df1, df2)
    return tr


def tcr_test():
    config.set_config(config.speciesType.human, config.chainType.alpha)
    data = load_data()
    data = compute_count(data, config.get_columns())
    print(data.iloc[:200, :])
    tr = compute_distance(df1=data.iloc[:200, :], df2=data)
    print(tr)
    # print(tr.rw_alpha)
    # print(tr.rw_alpha.shape)


def gliph_test():
    # TODO
    return


def giana_test():
    config.set_config(config.speciesType.human, config.chainType.beta)
    config.set_method(config.methodType.giana)
    data = load_data(fineCut=True).iloc[:200, :]
    data = compute_count(data, config.get_columns())
    # We need to rename the count as required by Giana
    data = data.rename(columns={'count': 'count..templates.reads.'})
    data = tcr_drop(data, 8)
    data = data.reset_index(drop=True)
    print(data.shape)
    distance_matrix = compute_distance(data)
    print(distance_matrix)



if __name__ == "__main__":
    giana_test()
