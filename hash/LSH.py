from . import *
from .Signature import MakeSignature
import numpy as np
from collections import defaultdict
import scipy.sparse as sps
import matplotlib.pyplot as plt

class LSH:

    def __init__(self, sig, num_bands=10):
        self.num_bands = num_bands
        self.sig = sig
        self.buckets = []
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

    profile = False

    def test_cosine_hash():

        from scipy.spatial.transform import Rotation as R

        N = 5000
        M = 3
        Hpp = 300
        NumBands = np.array([5, 10, 20, 30, 50, 100])

        ref = np.random.normal(size=(M,))
        ref /= np.linalg.norm(ref)

        for b, B in enumerate(NumBands):

            ref2 = np.random.normal(size=(M,N))
            C = np.cross(ref, ref2, axisb=0)
            C = np.divide(C, np.expand_dims(np.linalg.norm(C, axis=1), axis=1))
            C = np.multiply(C, np.random.uniform(0, np.pi,size=(N,1)))
            rot = R.from_rotvec(C)
            X = sps.csc_matrix(rot.apply(ref).transpose())

            csh = MakeSignature('CosineHash', num_row=3, num_hpp=Hpp)
            lsh = LSH(csh, num_bands=B)
            lsh.insert(X)

            sim_set = lsh.find_similar(sps.csc_matrix(np.expand_dims(ref, axis=1), shape=(M,1)))
            Y = np.arccos(ref * X[:, list(sim_set)])
            if not profile:
                plt.subplot(len(NumBands),1, b+1)
                plt.hist(Y, bins=np.linspace(0, np.pi, 50))

        if not profile:
            plt.show()


    if not profile:
        test_cosine_hash()
    else:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()
        pr.run('test_cosine_hash()')
        pr.disable()
        sortby = 'cumulative'
        ps = pstats.Stats(pr).sort_stats(sortby)
        ps.print_stats()