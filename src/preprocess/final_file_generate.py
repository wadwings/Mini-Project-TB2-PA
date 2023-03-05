import pandas as pd


def generate_final_file(file_path='./file.tsv', save_path='./final_file.tsv'):
    df = pd.read_table(file_path).loc[:, ['CDR3b', 'TRBV', 'TRBJ', 'CDR3a', 'TRAV', 'TRAJ', 'meta.subject.id']]
    df['TRBV'] = df['TRBV'].str.replace(r'\*01', '', regex=True)
    df['TRBJ'] = df['TRBJ'].str.replace(r'\*01', '', regex=True)
    df['TRAV'] = df['TRAV'].str.replace(r'\*01', '', regex=True)
    df['TRAJ'] = df['TRAJ'].str.replace(r'\*01', '', regex=True)
    df = df.rename(columns={'meta.subject.id': 'Patient'})
    df_count = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'Counts'})
    df = df.drop_duplicates().merge(df_count, on=df.columns.tolist(), how='left')
    df['Counts'] = df['Counts'].fillna(1)
    # df = df.fillna('')
    df['Counts'] = df['Counts'].astype(int)
    df.to_csv(save_path, sep='\t', index=False)

if __name__ == '__main__' :
    generate_final_file('./file.tsv')


