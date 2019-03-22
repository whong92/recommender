from . import *
import numpy as np
from .Signature import MinHashSignature
import scipy.sparse as sps
import matplotlib.pyplot as plt
from scipy.spatial.distance import jaccard


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