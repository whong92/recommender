from . import *
import numpy as np
from collections import defaultdict
import json
from zlib import adler32
#import os, sys
#sys.path.append(os.path.abspath('..'))
from ..utils.mongodbutils import DataService
import pymongo as pm
from bson.objectid import ObjectId

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
        self.features = {}

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
            Xindex = np.arange(0, M.shape[1], dtype=int)
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

        for i in range(X.shape[1]):
            self.features[int(Xindex[i])] = X[:,i]

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


class LSHDB(LSH):

    def __init__(self, sig, data_service=None, pref=None, num_bands=10, path=None):
        self.sig = sig
        if path is not None:
            with open(path, 'r', encoding='utf-8') as fp:
                conf = json.load(fp)
                self.num_bands = conf['num_bands']
                self.pref = conf['pref']
                self.data = DataService(conf['conn'], conf['db_name'])
        else:
            self.num_bands = num_bands
            self.pref = pref
            self.data = data_service
        super(LSHDB, self).__init__(self.sig, self.num_bands, None)

    def insert(self, X, Xindex=None, flush_every=1000):
        n = flush_every
        if Xindex is None:
            Xindex = np.arange(0, X.shape[1], dtype=int)
        for i in range(int(np.ceil(X.shape[1]/n))):
            Xb = X[:,n*i:n*(i+1)]
            Xbindex = Xindex[n*i:n*(i+1)]
            super(LSHDB, self).insert(Xb, Xbindex)
            self.flush()

    def flush(self):
        # TODO: make this write atomic by using transactions
        # buckets
        for b, bucket in enumerate(self.buckets):
            self.data.update_band(self.pref, int(b), bucket)

        # empty buckets
        self.buckets = []
        for i in range(self.num_bands):
            self.buckets.append(defaultdict(set))

        # features
        self.data.insert_features(self.pref, self.features)
        self.features = {}

    def find_similar(self, X):

        M = self.sig.generate_signature(X)
        B = self.num_bands
        R = int(M.shape[0] / B)
        sim_set = set()
        for b in range(self.num_bands):
            H = np.apply_along_axis(
                lambda x : np.int64(adler32(x.tobytes()))%NUM_BUCKETS,
                axis=0,arr=M[b*R:(b+1)*R,:]
            )
            for i, h in enumerate(H):
                sim_set = sim_set.union(set(self.data.get_bucket(self.pref, b, h)))
        return sim_set


    def save(self, path=None):

        assert path is not None
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(
                {'num_bands': self.num_bands,
                 'pref': self.pref,
                 'conn': self.data.conn,
                 'db_name': self.data.db_name,
                 }
                , fp
            )
