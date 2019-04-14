from . import *
import numpy as np
from collections import defaultdict
import json
from zlib import adler32

def MakeLSH(name, *args, **kwargs):
    switcher = {
        'LSHSimple' : LSH,
        'LSHDB': LSHDB,
    }
    assert name in switcher, "{0} not in {1}".format(name, switcher.keys())
    return switcher[name](*args, **kwargs)

class LSHInterface:
    def __init__(self):
        pass

    def insert(self, X, Xindex=None):
        raise NotImplementedError

    def find_similar(self, X):
        raise NotImplementedError

    def save(self, path=None):
        raise NotImplementedError

class LSH(LSHInterface):

    def __init__(self, sig, num_bands=10, path=None):

        super(LSH, self).__init__()

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

    def insert(self, X, Xindex=None):

        M = self.sig.generate_signature(X)
        B = self.num_bands
        if Xindex is None:
            Xindex = np.linspace(0, M.shape[1], M.shape[1], dtype=int)
        else:
            assert Xindex.shape[0]==M.shape[1], Xindex.dtype is int

        assert B < M.shape[0] and M.shape[0]%B==0, \
            "number of buckets must divide number of rows! B = {0}, R = {1}".format(B, M.shape[0])
        R = int(M.shape[0]/B)

        for b,bucket in enumerate(self.buckets):
            H = np.apply_along_axis(
                lambda x : np.int64(adler32(x.tobytes()))%NUM_BUCKETS,
                axis=0,arr=M[b*R:(b+1)*R,:]
            )
            for i, h in enumerate(H):
                bucket[h].add(int(Xindex[i]))

        return self.buckets

    def find_similar(self, X):
        M = self.sig.generate_signature(X)
        B = self.num_bands
        R = int(M.shape[0] / B)
        sim_set = set()
        for b, bucket in enumerate(self.buckets):
            H = np.apply_along_axis(
                lambda x : np.int64(adler32(x.tobytes()))%NUM_BUCKETS,
                axis=0,arr=M[b*R:(b+1)*R,:]
            )

            for i, h in enumerate(H):
                sim_set = sim_set.union(bucket[h])
        return sim_set

    def save(self, path=None):

        assert path is not None

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
                lambda x: np.int64(adler32(x.tobytes())) % NUM_BUCKETS,
                axis=0, arr=M[b * R:(b + 1) * R, :]
            )
            for i, h in enumerate(H):
                sim_set[i] = sim_set[i].union(bucket[h])
        return sim_set


class LSHDB(LSHInterface):
    def __init__(self, dbconn=None):
        super(LSHDB, self).__init__()
        self.dbconn=dbconn