import pandas as pd

def readCsv():
    df = pd.read_csv('data/heart_2020_cleaned.csv')
    df = df.drop(['Race', 'DiffWalking', 'GenHealth'], axis=1)

    return df
