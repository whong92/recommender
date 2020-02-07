from .recommenderInterface import Recommender
from .MatrixFactor import MatrixFactorizer
from sklearn.metrics.pairwise import cosine_similarity
# from tensorflow import predictor
import numpy as np
import scipy.sparse as sps
import json
import tensorflow as tf
import os
import shutil
import glob

class RecommenderMF(Recommender):

    def __init__(self, mode='train'
                 # arguments for train mode
                 , n_users=None, n_items=None
                 , mf_kwargs=None
                 # arguments for predict mode
                 , model_path=None):

        super(RecommenderMF, self).__init__(model_path)

        self.mode = mode
        self.estimator = None
        self.predictor = None
        # only for training
        self.config = {
            'n_users': n_users, 'n_items': n_items
        }

        if mode is 'train':
            if mf_kwargs is None:
                mf_kwargs = {'f': 20, 'lamb': .001, 'lr': .01, 'decay': 0.0}
            self.estimator = self._get_model()(model_path, n_users, n_items, **mf_kwargs)

        if mode is 'predict':

            assert model_path is not None

            config_path = os.path.join(model_path, 'config.json')

            with open(config_path, 'r', encoding='utf-8') as fp:
                self.config = json.load(fp)

            #self.predictor = tf.saved_model.load(model_path)
            self.predictor = self._get_model()(
                os.path.join(model_path, 'model.h5')
                ,n_users, n_items, mode=mode
            )
            self.data = None
            self.input_format = None

    def _get_model(self):
        return MatrixFactorizer

    def input_data(self, ds_train, ds_test):
        self.data_train = ds_train
        self.data_test = ds_test

    def input_array_data(self, u_train, i_train, r_train, u_test, i_test, r_test):
        self.u_train = u_train
        self.i_train = i_train
        self.r_train = r_train
        self.u_test = u_test
        self.i_test = i_test
        self.r_test = r_test

    def train(self, numepochs=30, batchsize=5000):
        assert self.mode is 'train', "must be in train mode!"
        # self.estimator.fit(self.data_train, self.data_test, numepochs=numepochs, batchsize=batchsize)
        self.estimator.fit(
            self.u_train, self.i_train, self.r_train,
            self.u_test, self.i_test, self.r_test,
            numepochs=numepochs, batchsize=batchsize
        )

    def save(self, path):
        self.estimator.save()
        self._save_config(path)

    def _save_config(self, lsh_path):
        with open(os.path.join(lsh_path, 'config.json'), 'w', encoding='utf-8') as fp:
            json.dump(self.config, fp)

    def predict(self, u_in, i_in):
        assert self.mode is 'predict'
        bla = self.predictor.predict(u_in, i_in)
        bla = list(map(np.squeeze, bla))
        return {k:v for k,v in zip(['rhat', 'p', 'q'], bla)}

    def recommend(self, u_in):
        # manual prediction by enumerating all stuff
        p = self.predict(
            u_in=np.tile(np.array([u_in], dtype=np.int32), self.config['n_items']),
            i_in=np.arange(0, self.config['n_items'], dtype=np.int32))

        return np.argsort(p['rhat'])[::-1], np.sort(p['rhat'])[::-1]

    def similar_to(self, i_in):
        p = self.predict(
            u_in=np.tile(np.array([0], dtype=np.int32), self.config['n_items']),
            i_in=np.arange(0, self.config['n_items'], dtype=np.int32))
        q_in = p['q'][i_in]
        q = p['q']
        s = np.squeeze(cosine_similarity(q, np.expand_dims(q_in, axis=0)))

        return np.argsort(s)[::-1][1:], np.sort(s)[::-1][1:]






