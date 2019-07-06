import numpy as np
from .Signature import CosineSimSignature
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import scipy.sparse as sps

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
        csh = CosineSimSignature(num_hpp=Hpp)
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