from .recommenderInterface import Recommender
from .MatrixFactor import MatrixFactorizer, MatrixFactorizerBias
from ..hash.LSH import MakeLSH
from ..hash.Signature import MakeSignature
from ..utils.utils import csv2df, splitDf
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.contrib import predictor
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

            self.predictor = predictor.from_saved_model(model_path)

            self.data = None
            self.input_format = None

    def _get_model(self):
        return MatrixFactorizer

    def input_data(self, ds_train, ds_test):
        self.data_train = ds_train
        self.data_test = ds_test

    def train(self, numepochs=30, batchsize=5000):
        assert self.mode is 'train', "must be in train mode!"
        self.estimator.fit(self.data_train, self.data_test, numepochs=numepochs, batchsize=batchsize)

    def save(self, path):
        est_save = self.estimator.save(path)
        shutil.move(os.path.join(est_save.decode("utf-8"), "saved_model.pb"), path)
        shutil.move(os.path.join(est_save.decode("utf-8"), "variables"), path)
        self._save_config(path)

    def _save_config(self, lsh_path):
        with open(os.path.join(lsh_path, 'config.json'), 'w', encoding='utf-8') as fp:
            json.dump(self.config, fp)

    def predict(self, u_in, i_in):
        assert self.mode is 'predict'
        return self.predictor({'user': u_in, 'item': i_in})

    def recommend(self, u_in):
        # manual prediction by enumerating all stuff
        p = self.predictor({'user': np.tile(np.array([u_in], dtype=np.int32), self.config['n_items']),
                            'item': np.arange(0, self.config['n_items'], dtype=np.int32)})

        return np.argsort(p['rhat'])[::-1], np.sort(p['rhat'])[::-1]

    def similar_to(self, i_in):
        p = self.predictor({'user': np.tile(np.array([0], dtype=np.int32), self.config['n_items']),
                            'item': np.arange(0, self.config['n_items'], dtype=np.int32)})
        q_in = p['q'][i_in]
        q = p['q']
        print(q.shape, q_in.shape)
        s = np.squeeze(cosine_similarity(q, np.expand_dims(q_in, axis=0)))

        return np.argsort(s)[::-1][1:], np.sort(s)[::-1][1:]


class RecommenderMFBias(RecommenderMF):

    def _get_model(self):
        return MatrixFactorizerBias





