from abc import ABC, abstractmethod
import pandas as pd
from typing import Union, Iterable, List, Callable
from ..utils.utils import get_pos_ratings_padded, mean_nnz
import scipy.sparse as sps
import numpy as np
from pprint import pprint

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
    
class DictIterable:

    def __init__(self):
        self.upstream = None
    
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

class GetUserItems(DictIterable):

    def __init__(self, U: sps.csr_matrix):
        super(GetUserItems, self).__init__()
        self.U = U
        Ucsc = sps.csc_matrix(U)
        self.Bi = np.reshape(np.array(mean_nnz(Ucsc, axis=0, mu=0)), newshape=(-1,))

    def __iter__(self) -> (dict):
        assert self.upstream is not None
        U = self.U
        Bi = self.Bi
        for d in self.upstream:
            u = d['user']
            rs, ys = get_pos_ratings_padded(U, u, 0, offset_yp=1)
            bs = Bi[ys-1]
            e = {'user_item': ys, 'user_rating': rs, 'user_rating_bias': bs}
            e.update(d)
            yield e

def link(iters:List[Callable]):
    cur = None
    for it in iters: 
        if cur is None: cur = it
        else: cur = it(cur)
    return cur

class SparseMatRowIterator:

    def __init__(self, row_batch_size):
        self.row_batch_size = row_batch_size
        self.S = None

    def __call__(self, d:dict) -> Iterable:
        self.d = d
        assert 'S' in d and 'pad_val' in d
        return self

    def __iter__(self) -> dict:
        
        batch_size = self.row_batch_size
        d = self.d
        S = d['S']
        pad_val = d['pad_val']
        batch_size = self.row_batch_size
        rows = d['rows'] if 'rows' in d else np.arange(0, S.shape[0])
        num_iter = int(np.ceil(len(rows)/batch_size))
        
        for i in range(num_iter):
            s = i*batch_size
            e = min((i+1)*batch_size, len(rows))
            row = rows[s:e]
            val, col = get_pos_ratings_padded(S, row, pad_val, batchsize=batch_size)
            yield {'rows': row, 'cols': col, 'val': val}
        return

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

if __name__=="__main__":
    
    df = pd.DataFrame({
        'user': [1,2,3,4],
        'item': [1,2,3,4],
        'rating': [1.,2.,3.,4.],
    })
    rnorm = {'loc': 0.0, 'scale': 10.0}
    U = sps.csr_matrix((df['rating'], (df['user'], df['item'])))

    it = BasicDFDataIter(2)([df])
    nit = Normalizer({'rating': rnorm})(it)
    rename = Rename({'user': 'u_in', 'item': 'i_in', 'rating': 'rhat'})(nit)
    mfit = XyDataIterator(ykey='rhat')(rename)

    for rows in it: print(rows)
    for rows in it: print(rows)
    for rows in it: print(rows)

    for rows in nit: print(rows)
    for rows in nit: print(rows)
    for rows in nit: print(rows)

    for rows in mfit: print(rows)
    for rows in mfit: print(rows)
    for rows in mfit: print(rows)

    it = BasicDFDataIter(2)
    nit = Normalizer({'rating': rnorm})
    rename = Rename({'user': 'u_in', 'item': 'i_in', 'rating': 'rhat'})
    mfit = XyDataIterator(ykey='rhat')
    
    for rows in link([[df], it, nit, rename, mfit]): print(rows)

    it = BasicDFDataIter(2)
    itb = GetUserItems(U)
    nit = Normalizer({
        'rating': rnorm, 
        'user_rating': rnorm, 
        'user_rating_bias': rnorm
    })
    for rows in link([[df], it, itb, nit]): pprint(rows)

    for x in SparseMatRowIterator(2)({'S': U, 'pad_val': -1.}): print(x)