from .recommenderInterface import Recommender
from .MatrixFactor import MatrixFactorizer
from ..hash.LSH import MakeLSH
from ..hash.Signature import MakeSignature
from ..utils.utils import csv2df, splitDf
from tensorflow.contrib import predictor
import numpy as np
import scipy.sparse as sps
import json
import tensorflow as tf
import os


class recommenderMF(Recommender):

    def __init__(self, mode='train'
                 # arguments for train mode
                 , n_users=None, n_items=None
                 , signature_t='CosineHash', lsh_t='LSHSimple'
                 , mf_kwargs=None, signature_kwargs=None, lsh_kwargs=None
                 # arguments for predict mode
                 , model_path=None, lsh_path=None):

        super(recommenderMF, self).__init__(model_path)

        self.mode = mode
        self.estimator = None
        self.predictor = None
        # only for training
        self.config = {
            'n_users': n_users, 'n_items': n_items
            , 'signature_t': signature_t, 'lsh_t': lsh_t
        }
        self.lsh = None
        self.sig = None

        if mode is 'train':
            if mf_kwargs is None:
                mf_kwargs = {'f': 20, 'lamb': .001, 'lr': .01, 'decay': 0.0}
            self.estimator = MatrixFactorizer(n_users, n_items, **mf_kwargs)
            self._make_hash(signature_t, lsh_t, signature_kwargs, lsh_kwargs)

        if mode is 'predict':

            for x in [model_path, lsh_path]:
                assert x is not None

            config_path = os.path.join(lsh_path, 'config.json')
            sig_path = os.path.join(lsh_path, 'signature.json')
            hash_path = os.path.join(lsh_path, 'hash.json')

            with open(config_path, 'r', encoding='utf-8') as fp:
                self.config = json.load(fp)

            signature_t = self.config['signature_t']
            lsh_t = self.config['lsh_t']
            self.sig = MakeSignature(signature_t, path=sig_path)
            self.lsh = MakeLSH(lsh_t, self.sig, path=hash_path)
            self.predictor = predictor.from_saved_model(model_path)

    def train(self, u_train, i_train, r_train, u_test, i_test, r_test,
              model_path=None, lsh_path=None):

        assert self.mode is 'train', "must be in train mode!"

        self.estimator.fit(
            {'u_in': u_train, 'i_in': i_train}, r_train,
            {'u_in': u_test, 'i_in': i_test}, r_test,
            numepochs=30,
        )

        # generate embeddings for all items, and insert into LSH
        self._update_hash(None)
        if model_path is not None:
            self.estimator.save(model_path)
        if lsh_path is not None:
            self._save_hash(lsh_path)

    def _make_hash(self, signature_t='CosineHash', lsh_t='LSHSimple',
                   signature_kwargs=None, lsh_kwargs=None):
        if signature_kwargs is None:
            signature_kwargs = {'num_row': 20, 'num_hpp': 100}
        if lsh_kwargs is None:
            lsh_kwargs = {'num_bands':5}

        self.sig = MakeSignature(signature_t, **signature_kwargs)
        self.lsh = MakeLSH(lsh_t, self.sig, **lsh_kwargs)

    def _update_hash(self, items=None):
        # rebuild lsh
        if self.mode is 'train':
            # TODO: use predict mode in Matrix Factorizer
            pred = predictor.from_estimator(self.estimator.model, MatrixFactorizer._predict_input_fn)
        else:
            pred = self.predictor

        if items is None:
            items = np.arange(0, self.config['n_items'], dtype=np.int32)

        p = pred(
            {'u_in': np.zeros(shape=(len(items),), dtype=np.int32),
             'i_in': items
             })
        self.lsh.insert(sps.csc_matrix(p['q'].transpose()),Xindex=items)

    def _save_hash(self, lsh_path):
        with open(os.path.join(lsh_path, 'config.json'), 'w', encoding='utf-8') as fp:
            json.dump(self.config, fp)
        self.sig.save(os.path.join(lsh_path, 'signature.json'))
        self.lsh.save(os.path.join(lsh_path, 'hash.json'))

    def predict(self, u_in, i_in):

        assert self.mode is 'predict'
        return self.predictor({'u_in': u_in, 'i_in': i_in})

    def recommend_lsh(self, u_in):
        # recommendation using lsh
        assert type(u_in) is int

        p = self.predictor({'u_in': np.array([u_in], dtype=np.int32),
                        'i_in': np.array([0], dtype=np.int32)})

        bla = self.lsh.find_similar(
            sps.csc_matrix(np.expand_dims(p['p'][0,:], axis=1))
        )
        idx = np.array(list(bla), dtype=np.int32)
        p = self.predictor({'u_in': np.tile(np.array([u_in], dtype=np.int32), len(idx)),
                            'i_in': idx})
        return idx[np.argsort(p['rhat'])], np.sort(p['rhat'])

    def recommend(self, u_in):
        # manual prediction by enumerating all stuff
        p = self.predictor({'u_in': np.tile(np.array([u_in], dtype=np.int32), self.config['n_items']),
                            'i_in': np.arange(0, self.config['n_items'], dtype=np.int32)})

        return np.argsort(p['rhat']), np.sort(p['rhat'])











