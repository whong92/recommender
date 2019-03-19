from . import *
from .SignatureFactory import MakeSignature
import numpy as np
from collections import defaultdict
import scipy.sparse as sps

class LSH:

    def __init__(self, sig, num_bands=10):
        self.num_bands = num_bands
        self.sig = sig

    def generate_signature(self, X):

        M = self.sig.generate_signature(X, False)
        B = self.num_bands
        buckets = []
        for i in range(B):
            buckets.append(defaultdict(set))
        assert B < M.shape[0] and M.shape[0]%B==0, \
            "number of buckets must divide number of rows! B = {0}, R = {1}".format(B, M.shape[0])
        R = int(M.shape[0]/B)

        for b,bucket in enumerate(buckets):
            H = np.apply_along_axis(
                lambda x : np.int64(hash(str(x)))%NUM_BUCKETS,
                axis=0,arr=M[b*R:(b+1)*R,:]
            )
            for i, h in enumerate(H):
                bucket[h].add(i)

        return buckets


if __name__=="__main__":

    N = 500
    M = 1000
    H = 200
    B = 50

    NumElems = np.array([150000, 250000, 300000])

    for E in NumElems:

        rows = np.random.randint(0, M, E)
        cols = np.random.randint(0, N, E)
        data = np.ones(shape=(E,), dtype=int)
        X = sps.csc_matrix((data,(rows, cols)),shape=(M,N))
        msh = MakeSignature('MinHash', num_hash=H)
        lsh = LSH(msh, num_bands=B)
        buckets = lsh.generate_signature(X)

        num_collisions = 0
        for k in buckets[0]:
            if len(buckets[0][k]) > 1:
                num_collisions += 1

        print('number of non singleton buckets in first band: {0}, number of filled buckets : {1}'.format(num_collisions, len(buckets[0])))

    N = 5000
    M = 100
    Hpp = 300
    B = 10
    NumElems = np.array([1000, 10000, 20000, 50000, 250000])

    for E in NumElems:

        rows = np.random.randint(0, M, E)
        cols = np.random.randint(0, N, E)
        data = np.random.normal(0, 1.0, E)
        X = sps.csc_matrix((data, (rows, cols)), shape=(M, N))
        csh = MakeSignature('CosineHash', num_hpp=Hpp)

        lsh = LSH(csh, num_bands=B)
        buckets = lsh.generate_signature(X)

        num_collisions = 0
        for k in buckets[0]:
            if len(buckets[0][k]) > 1:
                num_collisions += 1

        print(
            'number of non singleton buckets in first band: {0}, '
            'number of filled buckets : {1}'.format(num_collisions,len(buckets[0])))