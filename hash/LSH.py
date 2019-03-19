from . import *
from .SignatureFactory import MakeSignature
import numpy as np
from collections import defaultdict
import scipy.sparse as sps
import matplotlib.pyplot as plt

class LSH:

    def __init__(self, sig, num_bands=10):
        self.num_bands = num_bands
        self.sig = sig

    def generate_signature(self, X):

        M = self.sig.generate_signature(X)
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

    def find_similar(self, X, buckets):
        M = self.sig.generate_signature(X)
        B = self.num_bands
        R = int(M.shape[0] / B)
        sim_set = set()
        for b, bucket in enumerate(buckets):
            H = np.apply_along_axis(
                lambda x : np.int64(hash(str(x)))%NUM_BUCKETS,
                axis=0,arr=M[b*R:(b+1)*R,:]
            )
            for i, h in enumerate(H):
                sim_set = sim_set.union(bucket[h])
        return sim_set


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
    NumBands = np.array([5, 10, 20, 30, 50, 100])
    E = 20000

    for B in NumBands:

        rows = np.random.randint(0, M, E)
        cols = np.random.randint(0, N, E)
        data = np.random.normal(0, 1.0, E)
        X = sps.csc_matrix((data, (rows, cols)), shape=(M, N))
        csh = MakeSignature('CosineHash', num_hpp=Hpp)

        lsh = LSH(csh, num_bands=B)
        buckets = lsh.generate_signature(X)

        num_collisions = 0
        bucket = buckets[0]
        for k in bucket:
            if len(bucket[k]) > 1:
                num_collisions += 1

        sim_set = lsh.find_similar(X[:,0], buckets)
        Y = X[:,0].transpose()*X[:,list(sim_set)]
        print(np.mean(np.abs(Y.todense())))
        print(len(sim_set))

        print(
            'number of non singleton buckets in first band: {0}, '
            'number of filled buckets : {1}'.format(num_collisions,len(buckets[0])))