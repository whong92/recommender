import pandas as pd
import numpy as np
import scipy.sparse as sps
import tensorflow as tf

def csv2df(file, item, user, rating, return_cat_mapping=False, **kwargs):

    df = pd.read_csv(file, usecols=[item, user, rating])
    df.rename(columns={
        item: 'item',
        user: 'user',
        rating: 'rating',
    }, inplace=True)

    df['item_cat'] = df['item'].astype("category").cat.codes
    df['user_cat'] = df['user'].astype("category").cat.codes
    df['rating'] = df['rating'].astype(np.float64)

    if return_cat_mapping:
        user_map = df[['user_cat', 'user']].drop_duplicates()
        item_map = df[['item_cat', 'item']].drop_duplicates()

    df['item'] = df['item_cat']
    df['user'] = df['user_cat']

    if return_cat_mapping:
        return df, user_map, item_map, np.unique(df['user']).shape[0], np.unique(df['item']).shape[0]

    return df, np.unique(df['user']).shape[0], np.unique(df['item']).shape[0]

def getCatMap(chunks):
    user_map = {}
    item_map = {}
    def populate_maps(x):
        x = int(x)
        if x not in user_map:
            user_map[x] = len(user_map)
        if x not in item_map:
            item_map[x] = len(item_map)
    for chunk in chunks:
        chunk.apply(
            lambda x: populate_maps(x['user']),
            axis=1
        )
    return user_map, item_map

def getChunk(file, item, user, rating, chunksize=1000):
    dfReader = pd.read_csv(file, usecols=[item,user,rating], iterator=True, chunksize=chunksize)
    for chunk in dfReader:
        chunk.rename(columns={
            item: 'item',
            user: 'user',
            rating: 'rating',
        }, inplace=True)
        yield chunk

def procChunk(chunk, user_map, item_map):
    user_map_df = pd.DataFrame.from_dict(
        user_map, orient='index', columns=['user_cat']
    )
    item_map_df = pd.DataFrame.from_dict(
        item_map, orient='index', columns=['item_cat']
    )
    chunk = chunk.merge(user_map_df, left_on='user', right_index=True)
    chunk = chunk.merge(item_map_df, left_on='item', right_index=True)
    return chunk

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
    test_df = df.iloc[perm[:int(len(df) * train_test_split)]]
    return train_df, test_df

def makeTfDataset(input, batchsize, numepochs):
    ds = tf.data.Dataset.from_tensor_slices(input)
    ds = ds.batch(batchsize)
    ds = ds.repeat(numepochs)
    iterator = ds.make_initializable_iterator()
    next = iterator.get_next()
    return iterator, next

def csv2TfRecords(in_filename, out_filename):
    df, N, M = csv2df(in_filename, out_filename)
