import numpy as np
import numpy.matlib as nm
import scipy.sparse as sps
from . import *

def MakeSignature(name, *args, **kwargs):
    switcher = {
        'MinHash' : MinHashSignature,
        'CosineHash': CosineSimSignature
    }
    assert name in switcher, "{0} not in {1}".format(name, switcher.keys())
    return switcher[name](*args, **kwargs)

class Signature:
    def __init__(self):
        self.hashes = []
        return

    def generate_signature(self, x):
        raise NotImplementedError

class CosineSimSignature(Signature):

    def __init__(self, num_hpp=300, num_row=100):
        super(CosineSimSignature, self).__init__()
        self.Nhpp = num_hpp
        self.num_row = num_row
        self.hpp = np.random.normal(0, 1.0, (self.num_row, self.Nhpp))

    def generate_signature(self, X):
        R = X.shape[0]
        assert R==self.num_row, "number of rows of X not equal to number of " \
                                "rows declared when initializing hash function"
        XT = X.transpose()
        XT = XT.tocsr()
        B = XT * self.hpp > 0
        return B.transpose()


class MinHashSignature(Signature):
    def __init__(self, num_hash=10):
        super(MinHashSignature, self).__init__()
        self.num_hash = num_hash
        self.hash_fn = None
        self.__make_hash__()

    def __make_hash__(self):
        S = NUM_BUCKETS
        c = HASH_PRIME
        num_hash = self.num_hash

        A = np.random.randint(1, S, size=num_hash, dtype=np.int64)
        co = np.linspace(0,num_hash-1,num_hash,endpoint=True, dtype=int)
        A = sps.csr_matrix((A,(co,co)))
        b = np.expand_dims(np.random.randint(1, S, size=num_hash, dtype=np.int64), axis=1)
        self.hash_fn = lambda x: (np.array(A*x) + b) % c

    def generate_signature(self, X, new_hash=True):
        X = X.astype(np.int64)
        C = X.shape[1]
        H = self.num_hash
        M = -1*np.ones(shape=(H, C),dtype=np.int64)
        for c in range(C):
            rows = np.array(X.getcol(c).nonzero()[0])
            rows = nm.repmat(rows, H, 1)
            hashed_rows = self.hash_fn(rows)
            if hashed_rows.shape[0] > 0 and hashed_rows.shape[1] > 0:
                M[:,c] = np.min(hashed_rows, axis=1)
        return M

