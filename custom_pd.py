import pandas as pd
import scipy.stats as scs
import pickle as pkl

dataset_pkl = open("./data/datasets", "rb")
datasets = pkl.load(dataset_pkl)

train, test, valid = datasets

def categories(series):
    return range(int(series.min()), int(series.max()) + 1)


def chi_square_of_df_cols(df, col1, col2):
    """
    Computes the chi-square statistic and it's p-value of column co11 and col2 of df 

    """
    df_col1, df_col2 = df[col1], df[col2]

    result = [[sum((df_col1 == cat1) & (df_col2 == cat2))
               for cat2 in categories(df_col2)]
              for cat1 in categories(df_col1)]
    a = scs.chi2_contingency(result)
   

    return a[0],[1]


