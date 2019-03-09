from ..recommenderInterface import Recommender
from ..utils.utils import csv2df, df2umCSR
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sps
import numpy as np

class RecommenderCFSimple(Recommender):
    def __init__(self, model_file=None):
        super(RecommenderCFSimple, self).__init__(model_file)
        self.similarity = None
        self.denseUM = None
        self.nusers = 0
        self.nitems = 0
        return

    def from_csv(self, csv, item, user, rating, train_test_split=0.8):
        df, self.nusers, self.nitems = csv2df(csv, item, user, rating)
        perm = np.random.permutation(len(df))
        um_train = df2umCSR(df.ix[:perm[int(len(df)*train_test_split)]], self.nitems, self.nusers)
        um_test = df2umCSR(df.ix[perm[int(len(df) * train_test_split):]], self.nitems, self.nusers)
        self.training_data = {
            'csr': sps.csr_matrix(um_train, shape=(self.nitems, self.nusers)),
            'csc': sps.csc_matrix(um_train, shape=(self.nitems, self.nusers))
        }
        self.validation_data = {
            'csr': sps.csr_matrix(um_test, shape=(self.nitems, self.nusers)),
            'csc': sps.csc_matrix(um_test, shape=(self.nitems, self.nusers))
        }

    def train(self):
        assert self.training_data is not None
        assert self.validation_data is not None
        self.similarity = cosine_similarity(self.training_data['csr']) # can be replaced by locality-sensitive hashing
        self.denseUM = np.zeros(shape=(self.nitems, self.nusers))

        # pre-computation, expensive!
        for u in range(self.nusers):
            # TODO: extract only k most similar
            similar_set = self.training_data['csc'][:,u].nonzero()[0] #row numbers
            ratings = self.training_data['csc'][similar_set,u].data
            Sij = self.similarity[:,similar_set]
            nzi = np.sum(Sij,axis=1) > 0
            self.denseUM[nzi,u] = np.matmul(Sij[nzi,:], ratings)/np.sum(Sij[nzi,:], axis=1)

        train_idx = self.training_data['csr'].nonzero()
        train_error = np.sqrt(
            np.mean(
                np.power(
                    self.predict(train_idx[0], train_idx[1]) - self.training_data['csc'][train_idx[0], train_idx[1]],2
                )
            )
        )
        return train_error

    def predict(self, users, items, **kwargs):
        assert self.denseUM is not None
        assert len(users) == len(items)
        return self.denseUM[users, items]

if __name__=="__main__":
    rcf = RecommenderCFSimple()
    rcf.from_csv('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv', 'userId', 'movieId', 'rating')
    print(rcf.train())
    print(rcf.denseUM[:20,:20])
