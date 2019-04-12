import numpy as np
import numpy.matlib as nm
import scipy.sparse as sps
import json
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

    def __init__(self, num_hpp=300, num_row=10, path=None):

        super(CosineSimSignature, self).__init__()
        if path is not None:
            with open(path, 'r', encoding='utf-8') as fp:
                conf = json.load(fp)
            self.Nhpp = conf['num_hpp']
            self.num_row = conf['num_row']
            self.hpp = np.array(conf['hpp'])
            return

        self.Nhpp = num_hpp
        self.num_row = num_row
        self.hpp = np.random.normal(0, 1.0, (self.num_row, self.Nhpp))

    def generate_signature(self, X):
        R = X.shape[0]
        assert R==self.num_row, "number of rows of X not equal to number of " \
                                "rows declared when initializing hash function"
        XT = X.transpose()
        XT = sps.csr_matrix(XT)
        B = XT * self.hpp > 0
        return B.transpose()

    def save(self, path):
        sig_save = {
            'num_hpp': self.Nhpp,
            'num_row': self.num_row,
            'hpp': self.hpp.tolist()
        }
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(sig_save, fp)



class MinHashSignature(Signature):
    def __init__(self, num_hash=10, path=None):

        super(MinHashSignature, self).__init__()

        self.hash_fn = None
        self.hash_config = None

        if path is not None:
            with open(path, 'r', encoding='utf-8') as fp:
                conf = json.load(fp)
            self.num_hash = conf['num_hash']
            S = conf['num_buckets']
            c = conf['hash_prime']
            A = np.array(conf['hash_A'])
            b = np.array(conf['hash_b'])
            self.__make_hash__(S, c, A, b)
            return

        self.num_hash = num_hash
        self.__make_hash__()

    def __make_hash__(self, S=None, c=None, A=None,b=None):

        num_hash = self.num_hash

        if S is None:
            S = NUM_BUCKETS
            c = HASH_PRIME
            A = np.random.randint(1, S, size=num_hash, dtype=np.int64)
            b = np.expand_dims(np.random.randint(1, S, size=num_hash, dtype=np.int64), axis=1)

        self.hash_config = {
            'num_buckets': S,
            'hash_prime': c,
            'hash_A': A.tolist(),
            'hash_b': b.tolist()
        }

        co = np.linspace(0,num_hash-1,num_hash,endpoint=True, dtype=int)
        A = sps.csr_matrix((A,(co,co)))

        self.hash_fn = lambda x: (np.array(A*x) + b) % c

    def save(self, path):
        sig_save = {'num_hash':self.num_hash}
        sig_save.update(self.hash_config)
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(sig_save, fp)


    def generate_signature(self, X):
        X = X.astype(np.int64)
        X = sps.csr_matrix(X)
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


if __name__=="__main__":

    sig = MakeSignature('CosineHash')
    sig.save('./bla.json')
    sig2 = MakeSignature('CosineHash', path='./bla.json')
    X = np.random.normal(0, 1.0, size=(10,1))
    assert np.array_equal(sig.generate_signature(X), sig2.generate_signature(X))

    import os
    os.remove('./bla.json')

    sig = MakeSignature('MinHash')
    sig.save('./bla.json')
    sig2 = MakeSignature('MinHash', path='./bla.json')
    X = np.array(np.random.normal(0, 1.0, size=(100, 3)) > 0, dtype=int)
    assert np.array_equal(sig.generate_signature(X), sig2.generate_signature(X))
    import os

    os.remove('./bla.json')
