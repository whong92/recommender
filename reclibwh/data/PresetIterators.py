from reclibwh.data.iterators import *
import scipy.sparse as sps
import pandas as pd

def AUC_data_iter_preset(U: sps.csr_matrix, batchsize=10, rows=None):
    if rows is None: rows = np.arange(U.shape[0])
    return SparseMatRowIterator(batchsize, padded=True, negative=False)({'S': U, 'pad_val': -1, 'rows': rows})

def ALS_data_iter_preset(U: sps.csr_matrix, batchsize=20, rows=None):
    return link([
        {'S': U, 'pad_val': U.shape[1], 'rows': rows},
        SparseMatRowIterator(batchsize, padded=True, negative=False),
        Rename({'rows': 'u_in', 'cols': 'i_in', 'val': 'rhat'}),
        XyDataIterator(ykey='rhat')
    ])

def LMF_data_iter_preset(U: sps.csr_matrix, batchsize=20):
    return link([
        {'S': U, 'pad_val': -1.},
        SparseMatRowIterator(batchsize, padded=False, negative=True),
        Rename({'rows': 'u_in', 'cols': 'i_in', 'val': 'rhat'}),
        XyDataIterator(ykey='rhat')
    ])

def MF_data_iter_preset(df: pd.DataFrame, rnorm=None, batchsize=200):

    return link([
        [df],
        BasicDFDataIter(batchsize),
        Normalizer({} if not rnorm else {'rating': rnorm}),
        Rename({'user': 'u_in', 'item': 'i_in', 'rating': 'rhat'}),
        XyDataIterator(ykey='rhat')
    ])

def MFAsym_data_iter_preset(df: pd.DataFrame, U: sps.csr_matrix, batchsize=200, rnorm=None, Bi: np.array=None):

    it = BasicDFDataIter(batchsize)
    add_rated_items = AddRatedItems(U)
    add_bias = AddBias(U, item_key='user_rated_items', pad_val=-1, Bi=Bi)

    def add_1_to_user_rated_items_fn(d):
        d['user_rated_items'] += 1
        return d

    add_1_to_user_rated_items = LambdaDictIterable(add_1_to_user_rated_items_fn)
    rename = Rename({
        'user': 'u_in',
        'item': 'i_in',
        'rating': 'rhat',
        'user_rated_items': 'uj_in',
        'user_rated_ratings': 'ruj_in',
        'bias': 'bj_in',
    })
    nit = Normalizer({} if not rnorm else {'rhat': rnorm, 'ruj_in': rnorm, 'bj_in': rnorm})
    mfit = XyDataIterator(ykey='rhat')

    return link([
        [df], it, add_rated_items, add_bias, add_1_to_user_rated_items, rename, nit, mfit
    ])