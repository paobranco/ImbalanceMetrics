import pandas as pd
import numpy as np

def reg():
    # sample dataframe
    df = pd.read_csv(
        "data_reg.csv"
    )

    y = df['y_true']
    y_pred = df['y_pred']

    return y,y_pred

def cla():
    # sample dataframe
    df = pd.read_csv(
        "data_cla.csv"
    )

    y = df['y_true']
    y_pred = df['y_pred']
    y_proba = df[['y_proba0','y_proba1']]
    y_proba = y_proba.values.tolist()


    return y, y_pred, np.asarray([np.array(xi) for xi in y_proba])