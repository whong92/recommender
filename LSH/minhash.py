import numpy as np
import numpy.matlib as nm
import scipy.sparse as sps
import matplotlib.pyplot as plt
from scipy.spatial.distance import jaccard

class Signature:
    def __init__(self):
        self.hashes = []
        return

    def generate_signature(self, x):
        raise NotImplementedError

class MinHashSignature(Signature):
    def __init__(self, num_hash=10, a0=10, a1=5000, b0=10, b1=5000):
        super(MinHashSignature, self).__init__()
        self.a0 = a0
        self.a1 = a1
        self.b0 = b0
        self.b1 = b1
        self.num_hash = num_hash

    def __make_hash__(self, rows):
        S = 2 ** 16 - 1
        c = 65537
        num_hash = self.num_hash
        a0 = self.a0
        a1 = self.a1
        b0 = self.b0
        b1 = self.b1

        A = np.random.randint(1, S, size=num_hash)
        A = np.diag(A)
        b = np.expand_dims(np.random.randint(1, S, size=num_hash), axis=1)
        self.hash_fn = lambda x: (np.matmul(A,x)+b)%c

    def generate_signature(self, X):
        R = X.shape[0]
        self.__make_hash__(R)
        C = X.shape[1]
        H = self.num_hash
        M = -1*np.ones(shape=(H, C),dtype=int)
        for c in range(C):
            rows = np.array(X.getcol(c).nonzero()[0])
            rows = nm.repmat(rows, H, 1)
            hashed_rows = self.hash_fn(rows)
            if hashed_rows.shape[0] > 0 and hashed_rows.shape[1] > 0:
                M[:,c] = np.min(hashed_rows, axis=1)
        return M

if __name__=="__main__":
    N = 1000
    M = 1000
    H = 100
    rows = np.random.randint(0, M, 100000)
    cols = np.random.randint(0, N, 100000)
    data = np.ones(shape=(100000,), dtype=int)
    X = sps.csc_matrix((data,(rows, cols)),shape=(M,N))
    M = MinHashSignature(num_hash=H)
    S = M.generate_signature(X)

    jac = np.zeros(shape=(N,))
    countMinHash = np.zeros(shape=(N,))

    for i in range(N):
        if i == 0:
            continue
        jac[i] = 1 - jaccard(np.array(X.getcol(0).todense())>0
                         , np.array(X.getcol(i).todense())>0
                         )
        countMinHash[i] = float(np.sum(np.array(S[:,0]==S[:,i]).astype(int)))/H
        print(jac[i], countMinHash[i])

    plt.plot(jac, countMinHash, 'o')
    plt.xlim([0,.5])
    plt.ylim([0, .5])
    plt.show()