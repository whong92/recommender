from ..recommenderInterface import Recommender
from ..utils.utils import csv2df, df2umCSR, rmse
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt

class RecommenderCFSimple(Recommender):
    def __init__(self, model_file=None):
        super(RecommenderCFSimple, self).__init__(model_file)
        self.S = None # similarity matrix
        self.Ud = None # dense Utility matrix
        self.N = 0
        self.M = 0
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

    def train(self):

        assert self.training_data is not None
        assert self.validation_data is not None

        Ucsr = self.training_data['csr']
        Ucsc = self.training_data['csc']
        Vcsr = self.validation_data['csr']
        M = self.M
        N = self.N
        self.S = cosine_similarity(Ucsr) # can be replaced by locality-sensitive hashing
        self.Ud = np.zeros(shape=(M, N))

        mu = np.array(np.mean(self.training_data['csr']))
        bx = np.squeeze(np.array(np.mean(Ucsc, axis=0))) - mu
        bi = np.squeeze(np.array(np.mean(Ucsr, axis=1))) - mu

        # pre-computation, expensive!
        for u in range(N):
            # TODO: extract only k most similar
            similar_set = Ucsc[:,u].nonzero()[0] #row numbers
            ratings = np.array(Ucsc[similar_set,u].todense())
            Sij = self.S[:,similar_set]
            nzi = np.sum(Sij,axis=1) > 0
            bxi = mu + bx[u] + bi[nzi]
            bxj = mu + bx[u] + bi[similar_set]
            self.Ud[nzi,u] = bxi + np.divide(np.matmul(Sij[nzi,:], ratings[:,0] - bxj),np.sum(Sij[nzi,:], axis=1))

        train_idx = Ucsr.nonzero()
        test_idx = Vcsr.nonzero()
        train_error = rmse(self.predict(train_idx[0], train_idx[1]), Ucsr[train_idx[0], train_idx[1]])
        test_error = rmse(self.predict(test_idx[0], test_idx[1]), Vcsr[test_idx[0], test_idx[1]])
        return train_error, test_error

    def predict(self, users, items, **kwargs):
        assert self.Ud is not None
        assert len(users) == len(items)
        return self.Ud[users, items]

if __name__=="__main__":
    rcf = RecommenderCFSimple()
    rcf.from_csv('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv', 'userId', 'movieId', 'rating')
    train_error, test_error = rcf.train()
    print("train error {0}, test error {1}".format(train_error, test_error))
