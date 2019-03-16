import numpy as np
import scipy.sparse as sps
from .minhash import Signature, MinHashSignature
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.spatial.distance import jaccard

class CosineSimHash(Signature):

    def __init__(self, num_hpp=300):
        self.num_hpp = num_hpp

    def generate_signature(self, X, new_hash=True):
        Nhpp = self.num_hpp
        R = X.shape[0]
        C = X.shape[1]
        B = np.zeros(shape=(Nhpp,C), dtype=bool)
        XT = X.transpose()
        XT = XT.tocsr()
        hpp = np.random.normal(0, 1.0, (R,Nhpp))
        B = XT * hpp > 0
        return B.transpose()


if __name__=="__main__":

    N = 5000
    M = 100
    Hpp = 300
    NumElems = np.array([10000, 20000, 50000])

    for E in NumElems:

        rows = np.random.randint(0, M, E)
        cols = np.random.randint(0, N, E)
        data = np.random.normal(0, 1.0, E)
        X = sps.csc_matrix((data,(rows, cols)),shape=(M,N))
        csh = CosineSimHash(num_hpp=Hpp)
        S = csh.generate_signature(X)

        XT = X.transpose()
        XT = XT.tocsr()

        cs = cosine_similarity(XT)
        countMinHash = np.zeros(shape=(N-1,))
        theta = np.zeros(shape=(N - 1,))

        for i in range(1,N):
            if i == 0:
                continue
            countMinHash[i-1] = float(np.sum(np.array(S[:,0]==S[:,i]).astype(int)))/Hpp
            theta[i-1] = 1 - np.arccos(cs[0,i])/np.pi

        plt.plot(theta, countMinHash, 'o', label='{}'.format(E))

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.show()