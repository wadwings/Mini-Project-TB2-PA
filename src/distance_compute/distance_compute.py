import os

import pandas as pd
import numpy as np
import csv

from src.preprocess import *
from .giana.GIANA4 import giana
from tcrdist.repertoire import TCRrep
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


def gliph_method(df1, df2):
    def levenshtein_distance(c1, c2):
        rows = len(c1) + 1
        cols = len(c2) + 1
        distance = [[0 for col in range(cols)] for row in range(rows)]

        for row in range(1, rows):
            distance[row][0] = row
        for col in range(1, cols):
            distance[0][col] = col
        for col in range(1, cols):
            for row in range(1, rows):
                if c1[row - 1] == c2[col - 1]:
                    cost = 0
                else:
                    cost = 1
                distance[row][col] = min(distance[row - 1][col] + 1,
                                         distance[row][col - 1] + 1,
                                         distance[row - 1][col - 1] + cost)
        return distance[rows - 1][cols - 1]

    elements = list(df1)
    distance_result = [[levenshtein_distance(c1, c2) for c2 in elements] for c1 in elements]
    distance_matrix = pd.DataFrame(distance_result, columns=elements, index=elements)
    return distance_matrix


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
        config.distanceMethodType.tcrdist: tcrdist_method,
        config.distanceMethodType.giana: giana_method,
        config.distanceMethodType.gliph: gliph_method,
        'default': tcrdist_method,
    }.get(case, 'default')


def do_distance_compute(df1, df2=None):
    if df2 is None:
        df2 = df1
    print("compute distance stage")
    return method_selection(config.distance_method)(df1, df2)


def compute_single_distance(df1: pd.Series, df2: pd.Series):
    df1 = df1.to_frame().T
    df2 = df2.to_frame().T
    tr = do_distance_compute(df1, df2)
    return tr


def tcr_test():
    config.set_config(config.speciesType.human, config.chainType.alpha)
    data = load_data().iloc[:200, :]
    data = tcr_preprocess(data)
    tr = do_distance_compute(df1=data, df2=data)
    print(tr)
    # print(tr.rw_alpha)
    # print(tr.rw_alpha.shape)


def gliph_test():
    config.set_distance_method(config.distanceMethodType.gliph)
    config.set_config(config.speciesType.human, config.chainType.alpha)
    data_alpha = load_data().iloc[:200, :]
    data_alpha = data_alpha.drop('index', axis=1)
    data_alpha = tcr_drop(data_alpha, 8)
    data_alpha = data_alpha.reset_index(drop=True)
    data_alpha = data_alpha['CDR3a']
    data_alpha = cdr3_3_split(data_alpha)
    print(data_alpha)

    config.set_config(config.speciesType.human, config.chainType.beta)
    data_beta = load_data().iloc[:200, :]
    data_beta = data_beta.drop('index', axis=1)
    data_beta = data_beta.reindex(columns=['CDR3b'] + [col for col in data_beta.columns if col != 'CDR3b'])
    data_beta = tcr_drop(data_beta, 8)
    data_beta = data_beta.reset_index(drop=True)
    data_beta = data_beta['CDR3b']
    data_beta = cdr3_3_split(data_beta)
    distance_matrix = do_distance_compute(data_beta)
    print(distance_matrix)


def giana_test():
    config.set_config(config.speciesType.human, config.chainType.beta)
    config.set_distance_method(config.distanceMethodType.giana)
    data = load_data().iloc[:200, :]
    print(data.shape)
    distance_matrix = do_distance_compute(data)
    print(distance_matrix)


if __name__ == "__main__":
    giana_test()
