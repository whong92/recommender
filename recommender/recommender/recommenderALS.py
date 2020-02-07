from .recommenderInterface import Recommender
from .ALS import ALS
from .ALSTF import ALSTF
import scipy.sparse as sps
import os
import json
from recommender.recommender.recommenderCFSimple import RecommenderCFSimple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecommenderALS(Recommender):

    def __init__(self, mode='train'
                 # arguments for train mode
                 , n_users=None, n_items=None
                 , als_kwargs=None
                 # arguments for predict mode
                 , model_path=None):

        super(RecommenderALS, self).__init__(model_path)

        self.mode = mode
        self.als = None
        # only for training
        self.config = {
            'n_users': n_users, 'n_items': n_items
        }

        if mode is 'train':
            if als_kwargs is None:
                als_kwargs = {'K':10, 'lamb':1e-06, 'alpha':40.}
            self.als = ALSTF(batchsize=100, mode=mode, model_path=model_path, N=n_users, M=n_items, **als_kwargs)
            # self.als = ALS(model_path=model_path, N=n_users, M=n_items, **als_kwargs)

        if mode is 'predict':

            assert model_path is not None

            config_path = os.path.join(model_path, 'config.json')

            with open(config_path, 'r', encoding='utf-8') as fp:
                self.config = json.load(fp)

            self.data = None
            self.input_format = None
            if als_kwargs is None:
                als_kwargs = {}
            self.als = ALSTF(batchsize=300, mode=mode, model_path=model_path, N=n_users, M=n_items, **als_kwargs)
            # self.als = ALS(mode='predict', model_path=model_path, N=n_users, M=n_items)

    def input_array_data(self, u_train, i_train, r_train, u_test, i_test, r_test):
        self.training_data = RecommenderCFSimple.to_sparse_matrices(u_train, i_train, r_train, self.als.M, self.als.N)
        self.validation_data = RecommenderCFSimple.to_sparse_matrices(u_test, i_test, r_test, self.als.M, self.als.N)

    def train(self, steps=10):
        assert self.mode is 'train', "must be in train mode!"
        return self.als.train(sps.csr_matrix(self.training_data['csr'].T), steps=steps)

    def save(self, path):
        self.als.save(path)
        self._save_config(path)

    def _save_config(self, lsh_path):
        with open(os.path.join(lsh_path, 'config.json'), 'w', encoding='utf-8') as fp:
            json.dump(self.config, fp)

    def predict(self, u_in, i_in):
        assert self.mode is 'predict'
        return {'rhat': np.sum(np.multiply(self.als.X[u_in], self.als.Y[i_in]), axis=1), 'q': self.als.Y[i_in]}

    def recommend(self, u_in):
        # manual prediction by enumerating all stuff
        X = self.als.X[u_in]
        Y = self.als.Y
        p = np.sum(np.multiply(X, Y), axis=1)

        return np.argsort(p)[::-1], np.sort(p)[::-1]

    def similar_to(self, i_in):
        y = self.als.Y[i_in]
        Y = self.als.Y
        s = np.squeeze(cosine_similarity(Y, np.expand_dims(y, axis=0)))

        return np.argsort(s)[::-1][1:], np.sort(s)[::-1][1:]