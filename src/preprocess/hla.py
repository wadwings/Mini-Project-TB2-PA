import pandas as pd
import re

def generate_hla_table(file_path='./file.tsv', save_path='./hla.table'):
    df = pd.read_table(file_path).loc[:,['meta.subject.id', 'mhc.a', 'mhc.b', 'meta.donor.MHC']]
    df = df.drop_duplicates()
    df = df.reset_index()
    df = df.drop(columns='index')
    df['mhc.b'] = df['mhc.b'].str.replace(r'HLA-|\*', '', regex=True)
    df['mhc.a'] = df['mhc.a'].str.replace(r'HLA-|\*', '', regex=True)
    df['meta.donor.MHC'] = df['meta.donor.MHC'].str.replace(r'HLA-|\*', '', regex=True)
    df['mhc.b'] = df['mhc.b'].replace('B2M','')
    df['meta.subject.id'] = df['meta.subject.id'].astype(str)
    df = df.fillna('')
    hla_table = ''
    for i in range(len(df)):
        t = set(re.split(',|;',df['meta.donor.MHC'][i]) + re.split(',|;',df['mhc.a'][i]) + re.split(',|;',df['mhc.b'][i]))
        t.discard('')
        t = list(t)
        t.insert(0, df['meta.subject.id'][i])
        s = '\t'.join(t)
        hla_table += s + '\n'

    with open(save_path, 'w') as f:
        f.write(hla_table)

if __name__ == '__main__' :
    generate_hla_table('./file.tsv')
