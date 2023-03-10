from src.preprocess.setup import config
from src.distance_compute.giana.GIANA4 import giana
import pandas as pd
import csv
import os

def features_extraction(df1):
    df1.to_csv("./file.tsv", sep='\t', index=False, quoting=csv.QUOTE_NONE)
    giana({
        'File': './file.tsv',
        'Mat': True,
    })
    feature_matrix = pd.read_table('./file--RotationEncodingBL62.txt_EncodingMatrix.txt', header=None)
    # os.remove('./file.tsv')
    # os.remove('./file--RotationEncodingBL62.txt_EncodingMatrix.txt')
    # os.remove('./file--RotationEncodingBL62.txt')
    # os.remove('./VgeneScores.txt')
    return feature_matrix




