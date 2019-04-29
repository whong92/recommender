from . import *
import numpy as np
from collections import defaultdict
import json
from zlib import adler32
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


class LSHDB2(LSH):

    def __init__(self, sig, num_bands=10, dbconn=None, db=None, path=None):
        super(LSHDB2, self).__init__(sig, num_bands, None)
        self.dbclient = pm.MongoClient(dbconn)
        self.db = self.dbclient[db]
        self.num_bands = num_bands
        self.sig = sig

    def insert(self, X, Xindex=None, flush_every=1000):
        n = flush_every
        if Xindex is None:
            Xindex = np.arange(0, X.shape[1], dtype=int)
        for i in range(int(np.ceil(X.shape[1]/n))):
            Xb = X[:,n*i:n*(i+1)]
            Xbindex = Xindex[n*i:n*(i+1)]
            super(LSHDB2, self).insert(Xb, Xbindex)
            self.flush()

    def flush(self):

        # buckets
        for b, bucket in enumerate(self.buckets):
            band_name = 'band{:03d}'.format(b)
            band = self.db[band_name]
            for h in bucket:
                band.update(
                    {'_id': int(h)},
                    {'$push': {'vals': {'$each': list(bucket[h])}}},
                    upsert=True,
                )

        # empty buckets
        self.buckets = []
        for i in range(self.num_bands):
            self.buckets.append(defaultdict(set))

        # features
        for f in self.features:
            band = self.db["features"]
            band.update(
                {'_id': int(f)},
                {'$set': {'feature': list(self.features[f])}},
                upsert=True,
            )

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
                band_name = 'band{:03d}'.format(b)
                band = self.db[band_name]
                bucket_cursor = band.find({'_id': int(h)})
                for bucket in bucket_cursor:
                        sim_set = sim_set.union(bucket['vals'])
        return sim_set


    def save(self, path=None):

        assert path is not None
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump({'num_bands': self.num_bands}, fp)

class LSHDB(LSHInterface):

    def __init__(self, sig, num_bands=10, dbconn=None, db=None, path=None):
        super(LSHDB, self).__init__()
        self.dbclient = pm.MongoClient(dbconn)
        self.db = self.dbclient[db]
        self.num_bands = num_bands
        self.sig = sig

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

        feature_store = 'features'
        for i in range(X.shape[1]):
            feats = self.db[feature_store]
            feats.update(
                    {'_id': int(Xindex[i])},
                    {'$set': {'feature': list(np.squeeze(np.array(X[:,i].todense())))}},
                    upsert=True,
                )

        for b in range(self.num_bands):
            band_name = 'band{:03d}'.format(b)
            band = self.db[band_name]

            H = np.apply_along_axis(
                lambda x: np.int64(adler32(x.tobytes())) % NUM_BUCKETS,
                axis=0, arr=M[b * R:(b + 1) * R, :]
            )

            for i, h in enumerate(H):
                bucket_cursor = band.find({'_id': int(h)})
                bucket_vals = []
                for bucket in bucket_cursor:
                    bucket_vals = bucket['vals']
                bucket_vals.append(int(Xindex[i]))
                band.update(
                    {'_id': int(h)},
                    {'$set': {'vals': bucket_vals}},
                    upsert=True,
                )

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
                band_name = 'band{:03d}'.format(b)
                band = self.db[band_name]
                bucket_cursor = band.find({'_id': int(h)})
                for bucket in bucket_cursor:
                    sim_set = sim_set.union(bucket['vals'])
        return sim_set

    def save(self, path=None):

        assert path is not None
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump({'num_bands': self.num_bands}, fp)