from .recommenderInterface import Recommender
from ..utils.utils import rmse, mean_nnz
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from ..utils.ItemMetadata import ExplicitDataFromCSV

# TODO: standardize somewhat the input format for train and test across all recommenders

class RecommenderCFSimple(Recommender):

    def __init__(self, mode='train', model_file=None, k=10, n_users=0, n_items=0):
        super(RecommenderCFSimple, self).__init__(model_file)
        self.S = None # similarity matrix
        self.Ud = None # dense Utility matrix
        self.N = n_users
        self.M = n_items
        self.k = k
        self.mu = None
        self.bx = None
        self.bi = None

        if mode=='predict':
            assert model_file is not None, 'requires a model file to exist in predict mode'
            self.S = np.load(os.path.join(model_file, 'S.npy'))
            self.bx = np.load(os.path.join(model_file, 'bx.npy'))
            self.bi = np.load(os.path.join(model_file, 'bi.npy'))
            if os.path.exists(os.path.join(model_file, 'Ud.npy')):
                self.Ud = np.load(os.path.join(model_file, 'Ud.npy'))
            with open(os.path.join(model_file, 'stuff.json'), 'r') as fp:
                stuff = json.load(fp)
            self.N = stuff['N']
            self.N = stuff['M']
            self.N = stuff['k']

        return

    @staticmethod
    def to_sparse_matrices(u, i, r, M=None, N=None):
        if M is None:
            M = np.unique(i).shape[0]
        if N is None:
            N = np.unique(u).shape[0]
        return {
            'csr': sps.csr_matrix((r, (i, u)), shape=(M,N)),
            'csc': sps.csc_matrix((r, (i, u)), shape=(M,N))
        }

    def input_data(self, data: ExplicitDataFromCSV):
        data_train, data_test = data.make_training_datasets(dtype='dense')
        self.training_data = RecommenderCFSimple.to_sparse_matrices(*data_train, self.M, self.N)
        self.validation_data = RecommenderCFSimple.to_sparse_matrices(*data_test, self.M, self.N)


    def _compute_similarity_matrix(self):
        Ucsr = self.training_data['csr']
        # TODO: experiment with weighting schemes from here: https://www.benfrederickson.com/distance-metrics/
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
        test_error = rmse(self.predict(test_idx[1], test_idx[0]), Vcsr[test_idx[0], test_idx[1]])
        test_errors = self.predict(test_idx[1], test_idx[0]) - np.array(Vcsr[test_idx[0], test_idx[1]])
        return test_error, test_errors

    def predict(self, users, items):

        assert len(users) == len(items)

        rpred = np.zeros(shape=(len(users),))
        from tqdm import tqdm
        if self.Ud is None:
            for x,(u,i) in tqdm(enumerate(zip(users, items))):
                rpred[x] = self._compute_weighted_score(u, i)
        else:
            rpred = self.Ud[items, users]

        return rpred

    def recommend(self, u_in):
        # manual prediction by enumerating all stuff
        if self.Ud is None:
            p = self._compute_weighted_score(u_in)
        else:
            p = self.Ud[:, u_in]

        return np.argsort(p)[::-1], np.sort(p)[::-1]

    def save(self, model_file):
        if not os.path.exists(model_file):
            os.mkdir(model_file)
        np.save(os.path.join(model_file, 'S'), self.S)
        np.save(os.path.join(model_file, 'bx'), self.bx)
        np.save(os.path.join(model_file, 'bi'), self.bi)
        if self.Ud is not None:
            np.save(os.path.join(model_file, 'Ud'), self.Ud)
        stuff = {'N': self.N, 'M': self.M, 'k': self.k}
        with open(os.path.join(model_file, 'stuff.json'), 'w') as fp:
            json.dump(stuff, fp)

    def similar_to(self, i_in):
        s = self.S[i_in, :]
        return np.argsort(s)[::-1][1:], np.sort(s)[::-1][1:]

def compute_tfidf_weighted(_Mcsc, _Mcsr):
    nc = _Mcsc.shape[1]
    idf = np.log(nc)-np.log(_Mcsc.getnnz(axis=1)+1e-06)
    idf_csr = sps.csr_matrix(np.expand_dims(idf, axis=1))
    Mcsr = _Mcsr.copy()
    Mcsr.data = np.log(1 + _Mcsr.data)
    Mcsr = Mcsr.multiply(idf_csr)
    Mcsc = sps.csc_matrix(Mcsr)
    return Mcsr, Mcsc

class RecommenderCF_TFIDF(RecommenderCFSimple):

    def __init__(self, *args, **kwargs):
        super(RecommenderCF_TFIDF, self).__init__(*args, **kwargs)
        self.tf = None
        self.idf = None

    def _compute_similarity_matrix(self):

        # compute idf
        Ucsr = self.training_data['csr']
        Ucsc = self.training_data['csc']
        _Ucsr, _Ucsc = compute_tfidf_weighted(Ucsr, Ucsc)

        self.S = cosine_similarity(_Ucsr)


if __name__=="__main__":

    def main(k):

        data_folder = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small'
        model_folder = 'D:\\PycharmProjects\\recommender\\recommender\\models'


        ratings_csv = os.path.join(data_folder, 'ratings.csv')
        movies_csv = os.path.join(data_folder, 'movies.csv')
        d = ExplicitDataFromCSV(False,
                                ratings_csv=ratings_csv, r_item_col='movieId', r_user_col='userId',
                                r_rating_col='rating',
                                metadata_csv=movies_csv, m_item_col='movieId')
        """
        ratings_csv = os.path.join(data_folder, 'ratings_sanitized.csv')
        user_map_csv = os.path.join(data_folder, 'user_map.csv')
        item_map_csv = os.path.join(data_folder, 'item_map.csv')
        md_csv = os.path.join(data_folder, 'metadata.csv')
        stats_csv = os.path.join(data_folder, 'stats.csv')
        d = ExplicitDataFromCSV(True,
                                ratings_csv=ratings_csv,
                                user_map_csv=user_map_csv,
                                item_map_csv=item_map_csv,
                                md_csv=md_csv,
                                stats_csv=stats_csv)
        """
        d.save(data_folder)

        # training
        train_test_split = 0.8
        ds_train, ds_test, D_train, D_test = d.make_training_datasets(train_test_split)
        D_train.to_csv(os.path.join(data_folder, 'ratings_train.csv'), index=False)
        D_test.to_csv(os.path.join(data_folder, 'ratings_test.csv'), index=False)

        print(len(D_test), len(D_train))

        rcf = RecommenderCF_TFIDF(k=k, n_users=d.N, n_items=d.M)
        rcf.input_array_data(
            np.array(D_train['user'], dtype=np.int32),
            np.array(D_train['item'], dtype=np.int32),
            np.array(D_train['rating'], dtype=np.float64),
            np.array(D_test['user'], dtype=np.int32),
            np.array(D_test['item'], dtype=np.int32),
            np.array(D_test['rating'], dtype=np.float64),
        )
        test_error, es = rcf.train(precompute_ud=True)
        rcf.save('D:/PycharmProjects/recommender/models/CF_TFIDF_2019_11_02')
        print("k : {0} , test error {1}".format(k, test_error))
        plt.semilogx(k, test_error, 'bo')

    num_experiments = 1
    for i in range(num_experiments):
        for k in [100]:
            main(k)

    plt.show()

