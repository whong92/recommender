from ..recommenderInterface import Recommender
from ..utils.utils import csv2df, df2umCSR, rmse
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt

class RecommenderCFSimple(Recommender):
    def __init__(self, model_file=None, k=10):
        super(RecommenderCFSimple, self).__init__(model_file)
        self.S = None # similarity matrix
        self.Ud = None # dense Utility matrix
        self.N = 0
        self.M = 0
        self.k = k
        self.mu = None
        self.bx = None
        self.bi = None
        return

    def from_csv(self, csv, item, user, rating, train_test_split=0.8):
        df, self.N, self.M = csv2df(csv, item, user, rating)
        perm = np.random.permutation(len(df))
        um_train = df2umCSR(df.iloc[perm[:int(len(df)*train_test_split)]], self.M, self.N)
        um_test = df2umCSR(df.iloc[perm[int(len(df) * train_test_split):]], self.M, self.N)
        self.training_data = {
            'csr': sps.csr_matrix(um_train, shape=(self.M, self.N)),
            'csc': sps.csc_matrix(um_train, shape=(self.M, self.N))
        }
        self.validation_data = {
            'csr': sps.csr_matrix(um_test, shape=(self.M, self.N)),
            'csc': sps.csc_matrix(um_test, shape=(self.M, self.N))
        }

    def train(self, precompute_ud = False):

        assert self.training_data is not None
        assert self.validation_data is not None

        Ucsr = self.training_data['csr']
        Ucsc = self.training_data['csc']
        Vcsr = self.validation_data['csr']
        M = self.M
        N = self.N
        k = self.k
        self.S = cosine_similarity(Ucsr) # can be replaced by locality-sensitive hashing

        mu = np.array(np.mean(Ucsr))
        bx = np.squeeze(np.array(np.mean(Ucsc, axis=0))) - mu
        bi = np.squeeze(np.array(np.mean(Ucsr, axis=1))) - mu

        self.mu = mu
        self.bx = bx
        self.bi = bi

        if precompute_ud: # pre-computation, expensive!
            self.Ud = np.zeros(shape=(M, N))
            for u in range(N):
                ss = Ucsc[:,u].nonzero()[0] #row numbers
                ratings = np.array(Ucsc[ss,u].todense())[:,0]
                bxi = mu + bx[u] + bi
                bxj = bxi[ss]
                if self.S[:, ss].shape[1] >= k:
                    kidx = np.argpartition(self.S[:, ss], k-1)[:,:k]
                    ratings = ratings[kidx]
                    Sij = np.partition(self.S[:, ss], k-1, axis=1)[:,:k]
                    bxj = bxj[kidx]
                    T = np.sum(np.multiply(Sij, ratings-bxj),axis=1)
                else:
                    Sij = self.S[:, ss]
                    T = np.matmul(Sij, ratings - bxj)
                nzi = np.sum(Sij,axis=1) > 0
                self.Ud[nzi,u] = bxi[nzi] + np.divide(T[nzi],np.sum(Sij[nzi,:], axis=1))
                self.Ud[~nzi, u] = bxi[~nzi]

        test_idx = Vcsr.nonzero()
        test_error = rmse(self.predict(test_idx[0], test_idx[1]), Vcsr[test_idx[0], test_idx[1]])
        test_errors = self.predict(test_idx[0], test_idx[1]) - np.array(Vcsr[test_idx[0], test_idx[1]])
        return test_error, test_errors

    def predict(self, items, users, **kwargs):

        assert len(users) == len(items)
        rpred = np.zeros(shape=(len(users),))

        if self.Ud is None:
            Ucsc = self.training_data['csc'] # fast for col slices
            k = self.k
            mu = self.mu
            bx = self.bx
            bi = self.bi
            for x,(u,i) in enumerate(zip(users, items)):
                ss = Ucsc[:,u].nonzero()[0]
                ratings = np.array(Ucsc[ss, u].todense())[:,0]
                bxi = mu + bx[u] + bi
                bxj = bxi[ss]
                if len(self.S[i, ss]) >= k:
                    kidx = np.argpartition(self.S[i, ss], k - 1)[:k]
                    Sij = self.S[i, ss][kidx]
                    ratings = ratings[kidx]
                    bxj = bxj[kidx]
                else:
                    Sij = self.S[i,ss]
                if np.sum(Sij) > 0:
                    rpred[x] = bxi[i] + np.divide(np.dot(Sij, ratings - bxj), np.sum(Sij))
                else:
                    rpred[x] = bxi[i]
        else:
            rpred = self.Ud[items, users]

        return rpred

if __name__=="__main__":
    for k in [10,20,50,100,150,200]:
        rcf = RecommenderCFSimple(k=k)
        rcf.from_csv('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv', 'userId', 'movieId', 'rating')
        test_error, es = rcf.train(precompute_ud=True)
        print("k : {0} , test error {1}".format(k, test_error))
        #print(plt.hist(es, bins=50))
        plt.show()
