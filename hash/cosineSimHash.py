import numpy as np
import scipy.sparse as sps
from .minhash import Signature, MinHashSignature, NUM_BUCKETS

class cosineSimHash(Signature):

    def __init__(self, num_hpp=300 , num_minhash=100):
        self.num_hpp = num_hpp
        self.msh = MinHashSignature(num_minhash)

    def generate_signature(self, X):
        Nhpp = self.num_hpp
        R = X.shape[0]
        C = X.shape[1]
        B = np.zeros(shape=(Nhpp,C), dtype=bool)
        XT = X.transpose()
        XT = XT.tocsr()
        for i in range(Nhpp):
            hpp = np.random.normal(0, 1.0, (R))
            B[i,:] = XT*hpp > 0