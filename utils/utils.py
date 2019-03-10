import pandas as pd
import numpy as np
import scipy.sparse as sps

def csv2df(file, item, user, rating, **kwargs):
    df = pd.read_csv(file, usecols=[item, user, rating])
    df.columns = ['item', 'user', 'rating']
    df['item'] = df['item'].astype("category").cat.codes
    df['user'] = df['user'].astype("category").cat.codes
    df['rating'] = df['rating'].astype(np.float64)
    return df, np.unique(df['user']).shape[0], np.unique(df['item']).shape[0]

def df2umCSR(df, M=None, N=None):
    data = np.array(df['rating'])
    row = np.array(df['item']).astype(int)
    col = np.array(df['user']).astype(int)

    if M is None:
        M = np.unique(row).shape[0]
    if N is None:
        N = np.unique(col).shape[0]

    umCSR = sps.csr_matrix(
        (data, (row, col)),
        shape=(M, N)
    )
    return umCSR

def rmse(x, y):
    return np.sqrt(np.mean(np.power(x-y, 2)))

def mean_nnz(M, axis=None, mu=0):
    c = np.squeeze(np.array(M.getnnz(axis=axis)))
    s = np.squeeze(np.array(np.sum(M, axis=axis)))
    if axis is None:
        if c>0:
            return s/c
        else:
            return mu
    m = np.ones(shape=c.shape[0])*mu
    m[c>0] = np.divide(s[c>0],c[c>0])
    return m