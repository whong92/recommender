from ..recommenderInterface import Recommender
from ..utils.utils import csv2df, df2umCSR, rmse, mean_nnz
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD

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

    def _compute_similarity_matrix(self):
        Ucsr = self.training_data['csr']
        self.S = cosine_similarity(Ucsr)

    def _compute_weighted_score(self, u, i=None):

        Ucsc = self.training_data['csc']

        k = self.k
        mu = self.mu
        bx = self.bx
        bi = self.bi
        ss = Ucsc[:, u].nonzero()[0]  # row numbers
        ratings = np.array(Ucsc[:, u].todense())[ss, 0]
        bxi = mu + bx[u] + bi
        bxj = bxi[ss]

        if i is None:
            S = self.S[:, ss]
        else:
            S = np.expand_dims(self.S[i, ss], axis=0)

        if len(ss) >= k:
            kidx = np.argpartition(S, S.shape[1] - k + 1)[:, -k:]
            ratings = ratings[kidx]
            Sij = np.partition(S, S.shape[1] - k + 1, axis=1)[:, -k:]
            bxj = bxj[kidx]
            T = np.sum(np.multiply(Sij, ratings - bxj), axis=1)
        else:
            Sij = S
            T = np.matmul(Sij, ratings - bxj)

        if i is not None:
            bxi = np.expand_dims(bxi[i], axis=0)

        nzi = np.sum(Sij, axis=1) > 0
        pred = np.zeros(shape=(S.shape[0],))
        pred[nzi] = bxi[nzi] + np.divide(T[nzi],np.sum(Sij[nzi,:], axis=1))
        pred[~nzi] = bxi[~nzi]
        return pred

    def train(self, precompute_ud = False):

        assert self.training_data is not None
        assert self.validation_data is not None

        Ucsr = self.training_data['csr']
        Ucsc = self.training_data['csc']
        Vcsr = self.validation_data['csr']
        M = self.M
        N = self.N
        k = self.k

        self._compute_similarity_matrix() # can be replaced by locality-sensitive hashing

        # compute linear model
        mu = np.array(mean_nnz(Ucsr))
        bx = mean_nnz(Ucsc, axis=0, mu=mu) - mu
        bi = mean_nnz(Ucsr, axis=1, mu=mu) - mu
        self.mu = mu
        self.bx = bx
        self.bi = bi

        from tqdm import tqdm

        if precompute_ud: # pre-computation, expensive!
            self.Ud = np.zeros(shape=(M, N))
            for u in tqdm(range(N)):
                self.Ud[:,u] = self._compute_weighted_score(u)

        test_idx = Vcsr.nonzero()
        test_error = rmse(self.predict(test_idx[0], test_idx[1]), Vcsr[test_idx[0], test_idx[1]])
        test_errors = self.predict(test_idx[0], test_idx[1]) - np.array(Vcsr[test_idx[0], test_idx[1]])
        return test_error, test_errors

    def predict(self, items, users, **kwargs):

        assert len(users) == len(items)

        rpred = np.zeros(shape=(len(users),))
        from tqdm import tqdm
        if self.Ud is None:
            for x,(u,i) in tqdm(enumerate(zip(users, items))):
                rpred[x] = self._compute_weighted_score(u, i)
        else:
            rpred = self.Ud[items, users]

        return rpred

if __name__=="__main__":
    def main(k):
        rcf = RecommenderCFSimple(k=k)
        rcf.from_csv('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv', 'userId', 'movieId', 'rating')
        test_error, es = rcf.train(precompute_ud=True)
        # print("k : {0} , test error {1}".format(k, test_error))
        # plt.semilogx(k, test_error, 'bo')


    import cProfile, pstats

    pr = cProfile.Profile()
    pr.enable()
    pr.run('main(50)')
    pr.disable()
    sortby = 'cumulative'
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(20)

    """
    num_experiments = 10
    for i in range(num_experiments):
        for k in [2, 5, 10, 20, 50, 100, 150, 200]:
                main(k)

    plt.show()
    """
