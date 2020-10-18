from abc import ABC, abstractmethod
import pandas as pd
from typing import Union, Iterable, List, Callable, Any
from ..utils.utils import get_pos_ratings_padded, mean_nnz
import scipy.sparse as sps
import numpy as np
from pprint import pprint
from reclibwh.utils.utils import get_pos_ratings, get_neg_ratings
from ..utils.ItemMetadata import ExplicitDataFromCSV

class BasicDFDataIter:

    """[
        most basic recommender data iterator, all it does is iterate over a dataframe
        in batches of rows, containing a 'user' 'item' and 'rating' column
    ]
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.df = None

    def __call__(self, df:Iterable[pd.DataFrame]) -> Iterable:
        self.df = df
        return self

    def __iter__(self) -> dict:
        assert self.df is not None
        batch_size = self.batch_size
        for df in self.df:
            keys = df.keys()
            num_iter = int(np.ceil(len(df)/self.batch_size))
            data = {key: np.array(df[key]) for key in keys}
            for i in range(num_iter):
                s = i*batch_size
                e = min((i+1)*batch_size, len(df))
                yield {k: d[s:e] for k,d in data.items()}
        return
    
    def __len__(self):
        return sum([int(np.ceil(len(df)/self.batch_size)) for df in self.df])
    
class DictIterable:

    def __init__(self):
        self.upstream = None
    
    def __len__(self):
        # call upstream until we hit a handler (chain of responsibility pattern, very handy!)
        return len(self.upstream)
    
    def __call__(self, upstream:Iterable[dict]) -> Iterable:
        self.upstream = upstream
        return self

class XyDataIterator(DictIterable):

    def __init__(self, ykey: str):
        super(XyDataIterator, self).__init__()
        self.ykey = ykey
    
    def __iter__(self) -> (dict, dict):
        assert self.upstream is not None
        ykey = self.ykey
        for d in self.upstream:
            y = {ykey: d.pop(ykey)}
            X = d
            yield X, y

class Normalizer(DictIterable):

    def __init__(self, norms: dict):
        self.norms = norms
    
    @staticmethod
    def _normalize(norm, unnormalized: np.array):
        return (unnormalized - norm['loc'])/norm['scale']
    
    def __iter__(self) -> dict:
        assert self.upstream is not None
        for d in self.upstream:
            yield {
                k: (Normalizer._normalize(self.norms[k], v) if k in self.norms else v) for k, v in d.items()
            }

class UserItemsNegativeSampling(DictIterable):
    """
    Given an upstream iterator of users, samples positive and negative interactions
    from the utility matrix. Right now its hardcoded to sample a similar amount of
    negatives and positives, but this can always be changed
    """

    def __init__(self, U: sps.csr_matrix):
        super(UserItemsNegativeSampling, self).__init__()
        self.U = U
        Ucsc = sps.csc_matrix(U)
        self.Bi = np.reshape(np.array(mean_nnz(Ucsc, axis=0, mu=0)), newshape=(-1,))

    def __iter__(self) -> (dict):
        assert self.upstream is not None
        U = self.U
        Bi = self.Bi
        M = U.shape[1]
        for d in self.upstream:
            us = d['user']
            up, yp, rp = get_pos_ratings(U, us, M)
            uup, nup = np.unique(up, return_counts=True)
            un, yn = get_neg_ratings(U, us, M, samples_per_user=nup)
            rn = np.zeros(shape=un.shape, dtype=float)
            us = np.expand_dims(np.concatenate([up, un]), axis=1)
            ys = np.expand_dims(np.concatenate([yp, yn]), axis=1)
            rs = np.expand_dims(np.concatenate([rp, rn]), axis=1)
            bs = Bi[ys]
            e = {'user': us, 'item': ys, 'rating': rs, 'rating_bias': bs}
            e.update(d)
            yield e

def link(iters:List[Callable]):
    cur = None
    for it in iters: 
        if cur is None: cur = it
        else: cur = it(cur)
    return cur

class SparseMatRowIterator:

    def __init__(self, row_batch_size, padded=True, negative=False):
        self.row_batch_size = row_batch_size
        self.S = None
        self.padded = padded
        self.negative = negative
        if padded and negative: raise NotImplementedError("padded negative sampling not implemented")

    def __len__(self):
        d = self.d
        S = self.S
        batch_size = self.row_batch_size
        rows = d.get('rows')
        if rows is None: rows = np.arange(0, S.shape[0])
        return int(np.ceil(len(rows)/batch_size))

    def __call__(self, d:dict) -> Iterable:
        self.d = d
        assert 'S' in d and 'pad_val' in d
        self.S = d['S']
        return self

    def __iter__(self) -> dict:

        d = self.d
        S = d['S']
        M = S.shape[1]
        pad_val = d['pad_val']
        batch_size = self.row_batch_size
        rows = d.get('rows')
        if rows is None: rows = np.arange(0, S.shape[0])
        num_iter = int(np.ceil(len(rows)/batch_size))

        for i in range(num_iter):

            s = i*batch_size
            e = min((i+1)*batch_size, len(rows))
            row = rows[s:e]

            if self.padded: # TODO: negative padding not implemented yet
                val, col = get_pos_ratings_padded(S, row, pad_val)
            else:
                row, col, val = get_pos_ratings(S, row, M)
                if self.negative:
                    rowup, nup = np.unique(row, return_counts=True)
                    rown, coln = get_neg_ratings(S, row, M, samples_per_user=nup)
                    valn = np.zeros(shape=rown.shape, dtype=float)
                    row = np.concatenate([row, rown])
                    col = np.concatenate([col, coln])
                    val = np.concatenate([val, valn])
                row = np.expand_dims(row, axis=1)
                col = np.expand_dims(col, axis=1)
                val = np.expand_dims(val, axis=1)
            yield {'rows': row, 'cols': col, 'val': val, 'pad_val': pad_val}

        return

class AddBias(DictIterable):

    def __init__(self, U: sps.csr_matrix, item_key: str='items', pad_val: int=-1):
        super(AddBias, self).__init__()
        self.U = U
        Ucsc = sps.csc_matrix(U)
        self.Bi = np.reshape(np.array(mean_nnz(Ucsc, axis=0, mu=0)), newshape=(-1,))
        self.item_key = item_key
        self.pad_val = pad_val

    def __iter__(self) -> (dict):
        assert self.upstream is not None
        Bi = self.Bi
        for d in self.upstream:
            cols = d[self.item_key]
            pad_val = self.pad_val
            assert len(cols.shape)==2
            bias = -1*np.ones(shape=cols.shape)
            for r in range(bias.shape[0]):
                pad_mask = cols[r] == pad_val
                bias[r][~pad_mask] = Bi[cols[r][~pad_mask]]
            e = {'bias': bias}
            e.update(d)
            yield e

class AddRatedItems(DictIterable):

    def __init__(self, U: sps.csr_matrix, user_key: str='user'):
        super(AddRatedItems, self).__init__()
        self.U = U
        self.user_key = user_key

    def __iter__(self) -> (dict):
        assert self.upstream is not None
        for d in self.upstream:
            rows = d[self.user_key]
            rs, ys = get_pos_ratings_padded(self.U, rows, -1, offset_yp=0) # TODO: check this, don't add same rating!
            e = {'user_rated_ratings': rs, 'user_rated_items': ys}
            e.update(d)
            yield e

class Rename(DictIterable):

    def __init__(self, mapper: dict):
        super(Rename, self).__init__()
        self.mapper = mapper
    
    def __iter__(self) -> (dict, dict):
        assert self.upstream is not None
        mapper = self.mapper
        for d in self.upstream:
            for (j, k) in mapper.items():  d[k] = d.pop(j)
            yield d

class EpochIterator(DictIterable):
    def __init__(self, epochs: int):
        self.epochs = epochs
    
    def __iter__(self) -> Any:
        for e in range(self.epochs):
            for d in self.upstream:
                yield d

if __name__=="__main__":

    data_folder = '/home/ong/personal/recommender/data/ml-latest-small-2'
    d = ExplicitDataFromCSV(True, data_folder=data_folder)
    df_train = d.get_ratings_split(0)
    df_test = d.get_ratings_split(1)

    Utrain, Utest = d.make_training_datasets(dtype='sparse')
    
    # df = pd.DataFrame({
    #     'user': [0,1,1,1,2,3,4],
    #     'item': [1,1,2,3,2,3,4],
    #     'rating': [1.,1.,2.,3.,2.,3.,4.],
    # })
    # rnorm = {'loc': 0.0, 'scale': 10.0}
    # U = sps.csr_matrix((df['rating'], (df['user'], df['item'])))
    #
    # it = BasicDFDataIter(2)([df])
    # nit = Normalizer({'rating': rnorm})(it)
    # rename = Rename({'user': 'u_in', 'item': 'i_in', 'rating': 'rhat'})(nit)
    # mfit = XyDataIterator(ykey='rhat')(rename)
    #
    # for rows in it: print(rows)
    # for rows in it: print(rows)
    # for rows in it: print(rows)
    #
    # for rows in nit: print(rows)
    # for rows in nit: print(rows)
    # for rows in nit: print(rows)
    #
    # for rows in mfit: print(rows)
    # for rows in mfit: print(rows)
    # for rows in mfit: print(rows)
    #
    # it = BasicDFDataIter(2)
    # add_rated_items = AddRatedItems(U)
    # add_bias = AddBias(U, item_key='user_rated_items', pad_val=-1)
    # rename = Rename({
    #     'user': 'u_in',
    #     'item': 'i_in',
    #     'rating': 'rhat',
    #     'user_rated_items': 'uj_in',
    #     'user_rated_ratings': 'ruj_in',
    #     'bias': 'bj_in',
    # })
    # nit = Normalizer({'rhat': rnorm, 'ruj_in': rnorm, 'bj_in': rnorm})
    # mfit = XyDataIterator(ykey='rhat')
    #
    # for rows in link([[df], it, add_rated_items, add_bias, rename, nit, mfit]): pprint(rows)

    # it = SparseMatRowIterator(2, padded=False, negative=True)
    # add_bias = AddBias(U)
    # add_rated_items = AddRatedItems(U)
    # rename = Rename({'rows': 'u_in', 'cols': 'i_in', 'val': 'rhat'})
    # nit = Normalizer({
    #     'rating': rnorm,
    #     'user_rating': rnorm,
    #     'user_rating_bias': rnorm
    # })
    # for rows in link([{'S': U, 'pad_val': -1.}, it, add_rated_items, add_bias, rename]): pprint(rows)

    # for x in SparseMatRowIterator(2)({'S': U, 'pad_val': -1.}): print(x)

    d = {'user': np.ones(shape=(10,), dtype=int)}
    add_rated_items = AddRatedItems(Utrain)
    add_bias = AddBias(Utrain, item_key='user_rated_items', pad_val=-1)

    for row in link([[d], add_rated_items, add_bias]):
        print(row)