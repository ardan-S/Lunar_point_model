import pandas as pd


def combine(df1, df2, df3, df4):
    print("Combining not implemented, skipping...")
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    return df


def label(df):
    print("Labeling not implemented, skipping...")
    return df
