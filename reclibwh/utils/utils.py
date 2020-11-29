import pandas as pd
import numpy as np
import scipy.sparse as sps
import tensorflow as tf
from tqdm import tqdm
from typing import Iterable, Union


def csv2df(file, item, user, rating):
    df = pd.read_csv(file, usecols=[item, user, rating])
    df.rename(columns={
        item: 'item',
        user: 'user',
        rating: 'rating',
    }, inplace=True)
    return df


def normalizeDf(rating_df):
    df = rating_df.copy()

    df['item_cat'] = df['item'].astype("category").cat.codes
    df['user_cat'] = df['user'].astype("category").cat.codes
    df['rating'] = df['rating'].astype(np.float64)

    user_map = df[['user_cat', 'user']].drop_duplicates()
    item_map = df[['item_cat', 'item']].drop_duplicates()
    item_map.set_index('item', inplace=True)
    user_map.set_index('user', inplace=True)
    df.drop(['user', 'item'], inplace=True, axis=1)
    df.rename({'item_cat':'item', 'user_cat':'user'}, inplace=True, axis=1)

    return df, user_map, item_map

def getCatMap(chunks):

    user_map = {}
    item_map = {}
    tot_len = 0

    def populate_maps(x):
        u = int(x['user'])
        i = int(x['item'])
        if u not in user_map:
            user_map[u] = len(user_map)
        if i not in item_map:
            item_map[i] = len(item_map)

    for chunk in tqdm(chunks):

        chunk.apply(
            lambda x: populate_maps(x),
            axis=1
        )
        tot_len += len(chunk)
    return user_map, item_map, tot_len

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

def splitDf(df, train_test_split=0.8, return_df=False):
    assert train_test_split > 0
    perm = np.random.permutation(len(df))
    train_split = perm[:int(len(df) * train_test_split)]
    test_split = perm[int(len(df) * train_test_split):]
    if return_df:
        return df.iloc[train_split], df.iloc[test_split]
    return train_split, test_split

def sample_neg(pos:np.array, M:int, s:int):
    neg = np.zeros(shape=(0,), dtype=int)
    while(neg.shape[0]==0):
        neg = np.random.choice(M, size=min(s,M), replace=False).astype(int)
        neg = neg[np.in1d(neg, pos, assume_unique=True, invert=True)]
        break
    return neg

def get_neg_ratings(R: sps.csr_matrix, users:Iterable[int], M:int, samples_per_user:Union[np.ndarray, int]=50):
    """
    fetches negative ratings from a utility matrix R for given an array of users
    :param R:
    :param users:
    :param M:
    :param samples_per_user:
    :return: array of users and negative items
    """

    ru = R[users, :]
    if type(samples_per_user) is int:
        samples_per_user = np.ones(shape=(users.shape[0],), dtype=int)*samples_per_user
    ns = np.sum(samples_per_user)
    up = np.zeros(ns, dtype=int)
    yp = np.zeros(ns, dtype=int)

    offs = 0
    for u, (user, s) in enumerate(zip(users, samples_per_user)):
        neg = sample_neg(ru[u].indices, M, s)
        up[offs: offs+neg.shape[0]] = user
        yp[offs: offs+neg.shape[0]] = neg
        offs += neg.shape[0]

    return up[:offs], yp[:offs]

def get_pos_ratings(R: sps.csr_matrix, users:Iterable[int], M:int, batchsize=None):
    """
    fetches positive ratings and their values from a sparse utility matrix R for a given array of users
    :param R:
    :param users:
    :param M:
    :param batchsize:
    :return: users, items, ratings/interactions
    """
    ru = R[users, :]
    l = np.max(ru.getnnz(axis=1))
    if batchsize is None:
        batchsize = len(users)
    up = np.zeros(shape=(batchsize*l), dtype=np.int)
    rp = np.zeros(shape=(batchsize*l), dtype=np.float)
    yp = M*np.ones(shape=(batchsize*l), dtype=np.int)
    offs = 0
    for u, user in enumerate(users):
        numy = ru[u].data.shape[0]
        up[offs: offs+numy] = user
        rp[offs: offs+numy] = ru[u].data
        yp[offs: offs+numy] = ru[u].indices
        offs += numy

    return up[:offs], yp[:offs], rp[:offs]

def get_pos_ratings_padded(R, users, padding_val, batchsize=None, offset_yp=0, repeat_each=1, return_counts=False):
    """[summary]

    Arguments:
        R {[type]} -- [description]
        users {[type]} -- [description]
        padding_val {[type]} -- [description]

    Keyword Arguments:
        batchsize {[type]} -- [description] (default: {None})
        offset_yp {int} -- [description] (default: {0})
        repeat_each {int} -- [repeat the ratings for each user, faster than duplicating users] (default: {1})

    Returns:
        [type] -- [description]
    """

    ru = R[users, :]
    l = np.max(ru.getnnz(axis=1))
    if batchsize is None:
        batchsize = len(users)
    rp = np.zeros(shape=(batchsize*repeat_each, l), dtype=np.float)
    yp = int(padding_val)*np.ones(shape=(batchsize*repeat_each, l), dtype=np.int)
    counts = np.zeros(shape=(batchsize*repeat_each), dtype=np.int)

    for u, user in enumerate(users):
        rp[u*repeat_each:(u+1)*repeat_each, :ru[u].data.shape[0]] = np.repeat(np.expand_dims(ru[u].data, axis=0), repeat_each, axis=0)
        yp[u*repeat_each:(u+1)*repeat_each, :ru[u].indices.shape[0]] = np.repeat(np.expand_dims(ru[u].indices + offset_yp, axis=0), repeat_each, axis=0)
        counts[u*repeat_each:(u+1)*repeat_each] = len(ru[u].data)

    if return_counts: return rp, yp, counts
    return rp, yp
