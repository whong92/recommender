import pandas as pd
import numpy as np
import scipy.sparse as sps
import tensorflow as tf
from tqdm import tqdm


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

def getChunk(file, item, user, rating, chunksize=1000):
    dfReader = pd.read_csv(file, usecols=[item,user,rating], iterator=True, chunksize=chunksize)
    for chunk in dfReader:
        chunk.rename(columns={
            item: 'item',
            user: 'user',
            rating: 'rating',
        }, inplace=True)
        yield chunk

def procChunk(chunk, user_map_df, item_map_df):
    chunk = chunk.merge(user_map_df, left_on='user', right_index=True)
    chunk = chunk.merge(item_map_df, left_on='item', right_index=True)
    chunk['user'] = chunk['user_cat']
    chunk['item'] = chunk['item_cat']
    return chunk

def procSingleRow(row, user_map, item_map):
    row['user'] = user_map[int(row.iloc[0]['user'])]
    row['item'] = item_map[int(row.iloc[0]['item'])]
    return row

def tf_serialize_example(user, item, rating):
    tf_string = tf.py_func(
        serializeExample,
        (user,item,rating),
        tf.string
    )
    return tf.reshape(tf_string, ())

def serializeExample(user, item, rating):
    feature = {
        'user': tf.train.Feature(int64_list=tf.train.Int64List(value=[user])),
        'item': tf.train.Feature(int64_list=tf.train.Int64List(value=[item])),
        'rating': tf.train.Feature(float_list=tf.train.FloatList(value=[rating])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

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

def splitDf(df, train_test_split=0.8):
    assert train_test_split > 0
    perm = np.random.permutation(len(df))
    train_df = df.iloc[perm[:int(len(df) * train_test_split)]]
    test_df = df.iloc[perm[int(len(df) * train_test_split):]]
    return train_df, test_df

