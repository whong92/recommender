from sklearn.metrics.pairwise import cosine_similarity
from hash.Signature import MakeSignature
from hash.LSH import LSH
from .recommenderCFSimple import RecommenderCFSimple
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sps
import numpy as np

class RecommenderCFReduced(RecommenderCFSimple):

    def __init__(self, model_file=None, k=10, comp=50, use_lsh=False, num_hpp=300, num_lsh_bands=10):
        super(RecommenderCFReduced, self).__init__(model_file, k)
        self.comp = comp

        self.use_lsh = use_lsh
        if use_lsh:
            csig = MakeSignature("CosineHash", num_hpp=num_hpp, num_row=self.comp)
            self.lsh = LSH(csig, num_lsh_bands)
        return

    def _compute_similarity_matrix(self):

        Ucsr = self.training_data['csr']
        tsvd = TruncatedSVD(n_components=self.comp)
        tsvd.fit(Ucsr)
        Ucsr_red = sps.csr_matrix(tsvd.fit_transform(Ucsr))
        self.training_data['csr_red'] = Ucsr_red
        self.training_data['csc_red'] = sps.csc_matrix(Ucsr_red)
        self.S = (cosine_similarity(Ucsr_red) + 1) / 2

        if not self.use_lsh :
            self.S = (cosine_similarity(Ucsr_red) + 1) / 2  # can be replaced by locality-sensitive hashing
        else:
            self.lsh.insert(Ucsr_red.todense().transpose())

    def _compute_weighted_score(self, u, i=None):

        if not self.use_lsh:
            return super(RecommenderCFReduced, self)._compute_weighted_score(u, i)
        else:
            Ucsc_redT = self.training_data['csc_red'].transpose()
            Ucsc = self.training_data['csc']
            mu = self.mu
            bx = self.bx
            bi = self.bi
            ratings = np.array(Ucsc[:, u].todense())
            bxi = mu + bx[u] + bi
            ss = set(Ucsc[:, u].nonzero()[0])  # row numbers

            if i is None:
                cols = np.arange(0,Ucsc.shape[0],1)
                S = Ucsc_redT
                T = np.zeros(shape=(cols.shape[0]))
                kidxs = self.lsh.find_similar_multiple(S)
                for c in cols:
                    kidx = list(kidxs[c])
                    if len(kidx) > 0:
                        r = ratings[kidx]
                        T[c] = np.mean(r - bxi[kidx]) + bxi[c]
                    else:
                        T[c] = bxi[c]
                return T
            else:
                kidx = list(ss.intersection(self.lsh.find_similar(Ucsc_redT[:,i].todense())))
                if len(kidx) == 0:
                    return bxi[i]
                return np.mean(np.squeeze(ratings[kidx]) - np.squeeze(bxi[kidx])) + bxi[i]

if __name__=="__main__":
    def main(k, c):
        rcf = RecommenderCFReduced(k=k, comp=c)
        rcf.from_csv('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv', 'userId', 'movieId', 'rating')
        test_error, es = rcf.train(precompute_ud=True)
        print("c : {0} , test error {1}".format(c, test_error))
        plt.semilogx(c, test_error, 'bo')


    """
    import cProfile, pstats

    pr = cProfile.Profile()
    pr.enable()
    pr.run('main(50)')
    pr.disable()
    sortby = 'cumulative'
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(20)


    num_experiments = 10
    for i in range(num_experiments):
        for k in [2, 5, 10, 20, 50, 100, 150, 200]:
                main(k)
    """

    num_experiments = 1
    k = 50

    """
    c = 20
    hpp = 300
    b = 20
    rcf = RecommenderCFReduced(k=k, comp=c, use_lsh=True, num_hpp=hpp, num_lsh_bands=b)
    rcf.from_csv('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv', 'userId', 'movieId', 'rating')
    test_error, es = rcf.train(precompute_ud=False)
    print("c : {0} , test error {1}".format(c, test_error))
    """

    for i in range(num_experiments):
        for c in [2, 5, 10, 20, 50, 100, 150, 200]:
            main(k, c)

    plt.show()


