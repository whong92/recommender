import numpy as np
import numpy.matlib as nm
import scipy.sparse as sps
import matplotlib.pyplot as plt
from scipy.spatial.distance import jaccard

NUM_BUCKETS = 2**32 - 1
HASH_PRIME = 4295297389

class Signature:
    def __init__(self):
        self.hashes = []
        return

    def generate_signature(self, x):
        raise NotImplementedError

class MinHashSignature(Signature):
    def __init__(self, num_hash=10):
        super(MinHashSignature, self).__init__()
        self.num_hash = num_hash
        self.hash_fn = None

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
        if new_hash or self.hash_fn is None:
            self.__make_hash__()
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

    N = 500
    M = 1000
    H = 300
    NumElems = np.array([1000, 5000, 10000, 50000, 100000, 250000])

    for E in NumElems:

        rows = np.random.randint(0, M, E)
        cols = np.random.randint(0, N, E)
        data = np.ones(shape=(E,), dtype=int)
        X = sps.csc_matrix((data,(rows, cols)),shape=(M,N))
        mhs = MinHashSignature(num_hash=H)
        S = mhs.generate_signature(X)

        jac = np.zeros(shape=(N-1,))
        countMinHash = np.zeros(shape=(N-1,))

        k = 0
        for i in range(1,N):
            if i == 0:
                continue
            jac[i-1] = 1 - jaccard(np.array(X.getcol(0).todense())>0
                             , np.array(X.getcol(i).todense())>0
                             )
            countMinHash[i-1] = float(np.sum(np.array(S[:,0]==S[:,i]).astype(int)))/H
            k += 1

        #plt.plot(jac, countMinHash, 'o', label='{}'.format(E))
        plt.hist(jac)
        plt.show()
        print("Average jacc {}".format(np.mean(jac)))

    #plt.xlim([0,.5])
    #plt.ylim([0, .5])
    plt.legend()
    plt.show()