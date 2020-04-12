import json
import os

import numpy as np
import scipy.sparse as sps
from sklearn.metrics.pairwise import cosine_similarity

from recommender.core.ALS import ALSTF
from recommender.utils.ItemMetadata import ExplicitDataFromCSV
from .recommenderInterface import Recommender
from ..utils.eval_utils import AUCCallback
from typing import Iterable, Optional

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
        self.data = None
        # only for training
        self.config = {
            'n_users': n_users, 'n_items': n_items,
            'als_kwargs': als_kwargs,
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
            self.als = ALSTF(
                batchsize=300, mode=mode, model_path=os.path.join(model_path, 'epoch-best'),
                N=self.config['n_users'], M=self.config['n_items'], **self.config['als_kwargs'])
            # self.als = ALS(mode='predict', model_path=model_path, N=n_users, M=n_items)

    def input_data(self, data: ExplicitDataFromCSV):
        self.data = data
        # self.training_data, self.validation_data = data.make_training_datasets(dtype='sparse')
        return

    def add_users(self, num=1):
        self.als._add_users(num=num)

    def train(self):
        assert self.mode is 'train', "must be in train mode!"
        AUCC = AUCCallback(
            self.data, np.arange(0,self.data.N,self.data.N//300,dtype=int),
            save_fn=lambda: self.als.save_as_epoch('best')
        )
        AUCC.set_model(self)
        Utrain, _ = self.data.make_training_datasets(dtype='sparse')
        trace = self.als.train(Utrain, cb=AUCC)
        AUCC.save_result(os.path.join(self.model_file, 'AUC.csv'))
        return trace, AUCC.AUCe

    def train_update(self, users: Optional[Iterable[int]]=None):
        if users is None:
            users = np.arange(self.data.N)
        AUCC = AUCCallback(self.data, users, save_fn=lambda: self.als.save_as_epoch('best_updated'))
        AUCC.set_model(self)
        # TODO: improve so not required to construct the entire dataset for an update
        Utrain, _ = self.data.make_training_datasets(dtype='sparse')
        trace = self.als.train_update(Utrain, users, cb=AUCC, use_cache=True)
        AUCC.save_result(os.path.join(self.model_file, 'AUC_update.csv'))
        return trace, AUCC.AUCe

    def save(self, path):
        self.als.save(path)
        self._save_config(path)

    def _save_config(self, lsh_path):
        with open(os.path.join(lsh_path, 'config.json'), 'w', encoding='utf-8') as fp:
            json.dump(self.config, fp, indent=4)

    def predict(self, u_in, i_in):
        assert self.mode is 'predict'
        return {'rhat': np.sum(np.multiply(self.als.X[u_in], self.als.Y[i_in]), axis=1), 'q': self.als.Y[i_in]}

    def recommend(self, u_in):
        # manual prediction by enumerating all stuff
        X = self.als.X[u_in]
        Y = self.als.Y
        p = np.matmul(X, np.transpose(Y))  # np.sum(np.multiply(X, Y), axis=1)

        return np.argsort(p)[:, ::-1], np.sort(p)[:, ::-1]

    def similar_to(self, i_in):
        y = self.als.Y[i_in]
        Y = self.als.Y
        s = np.squeeze(cosine_similarity(Y, np.expand_dims(y, axis=0)))

        return np.argsort(s)[::-1][1:], np.sort(s)[::-1][1:]

    def recommend_similar_to(self, u_in, i_in):
        X = self.als.X[u_in]
        y = self.als.Y[i_in]
        Y = self.als.Y
        s = np.squeeze(cosine_similarity(Y, np.expand_dims(y, axis=0)))
        p = np.matmul(X, np.transpose(Y))  # np.sum(np.multiply(X, Y), axis=1)
        q = .5*(p + s)

        return np.argsort(q)[::-1], np.sort(q)[::-1]