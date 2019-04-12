from . import *
from .Signature import MakeSignature
import numpy as np
from collections import defaultdict
import scipy.sparse as sps
import matplotlib.pyplot as plt
import json

class LSH:

    def __init__(self, sig, num_bands=10, path=None):

        self.num_bands = num_bands
        self.sig = sig
        self.buckets = []

        if path is not None:
            with open(path, 'r', encoding='utf-8') as fp:
                conf = json.load(fp)
                buckets = conf['buckets']
                self.num_bands = conf['num_bands']
                for bucket in buckets:
                    self.buckets.append(defaultdict(set))
                    for k in bucket:
                        self.buckets[-1][int(k)] = self.buckets[-1][int(k)].union(set(bucket[k]))
            return

        for i in range(num_bands):
            self.buckets.append(defaultdict(set))

    def insert(self, X):

        M = self.sig.generate_signature(X)
        B = self.num_bands

        assert B < M.shape[0] and M.shape[0]%B==0, \
            "number of buckets must divide number of rows! B = {0}, R = {1}".format(B, M.shape[0])
        R = int(M.shape[0]/B)

        for b,bucket in enumerate(self.buckets):
            H = np.apply_along_axis(
                lambda x : np.int64(hash(x.tobytes()))%NUM_BUCKETS,
                axis=0,arr=M[b*R:(b+1)*R,:]
            )
            for i, h in enumerate(H):
                bucket[h].add(i)

        return self.buckets

    def find_similar(self, X):
        M = self.sig.generate_signature(X)
        B = self.num_bands
        R = int(M.shape[0] / B)
        sim_set = set()
        for b, bucket in enumerate(self.buckets):
            H = np.apply_along_axis(
                lambda x : np.int64(hash(x.tobytes()))%NUM_BUCKETS,
                axis=0,arr=M[b*R:(b+1)*R,:]
            )
            for i, h in enumerate(H):
                sim_set = sim_set.union(bucket[h])
        return sim_set

    def save(self, path):

        with open(path, 'w', encoding='utf-8') as fp:
            save_buckets = []
            for bucket in self.buckets:
                save_bucket = {}
                for k in bucket:
                    save_bucket[str(k)] = list(bucket[k])
                save_buckets.append(save_bucket)
            json.dump({'num_bands': self.num_bands, 'buckets': save_buckets}, fp)

    def find_similar_multiple(self, X):
        M = self.sig.generate_signature(X)

        B = self.num_bands
        R = int(M.shape[0] / B)
        sim_set = defaultdict(set)
        for b, bucket in enumerate(self.buckets):
            H = np.apply_along_axis(
                lambda x: np.int64(hash(x.tobytes())) % NUM_BUCKETS,
                axis=0, arr=M[b * R:(b + 1) * R, :]
            )
            for i, h in enumerate(H):
                sim_set[i] = sim_set[i].union(bucket[h])
        return sim_set


if __name__=="__main__":

    """
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
    """