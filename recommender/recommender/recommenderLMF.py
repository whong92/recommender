import json
import os

import numpy as np
import scipy.sparse as sps
from sklearn.metrics.pairwise import cosine_similarity

from recommender.core.LMF import LogisticMatrixFactorizer, LMFCallback
from recommender.utils.ItemMetadata import ExplicitDataFromCSV
from .recommenderInterface import Recommender
from ..utils.eval_utils import AUCCallback
from typing import Iterable, Optional


class RecommenderLMF(Recommender):

    def __init__(self, mode='train'
                 # arguments for train mode
                 , n_users=None, n_items=None
                 , lmf_kwargs=None
                 # arguments for predict mode
                 , model_path=None):

        super(RecommenderLMF, self).__init__(model_path)

        self.mode = mode
        self.lmf = None
        self.data = None
        # only for training
        self.config = {
            'n_users': n_users, 'n_items': n_items,
            'lmf_kwargs': lmf_kwargs,
        }

        if mode is 'train':
            if lmf_kwargs is None:
                lmf_kwargs = {'f':10, 'lamb':1e-06, 'alpha':40., 'lr':0.01, 'epochs':30, 'bias':False}
            self.lmf = LogisticMatrixFactorizer(batchsize=100, mode=mode, model_path=model_path, N=n_users, M=n_items, **lmf_kwargs)

        if mode is 'predict':

            assert model_path is not None

            config_path = os.path.join(model_path, 'config.json')

            with open(config_path, 'r', encoding='utf-8') as fp:
                self.config = json.load(fp)

            self.data = None
            self.input_format = None
            self.lmf = LogisticMatrixFactorizer(
                batchsize=300, mode=mode, model_path=os.path.join(model_path, "model-best.h5"),
                N=self.config['n_users'], M=self.config['n_items'], **self.config['lmf_kwargs']
            )

    def input_data(self, data: ExplicitDataFromCSV):
        self.data = data
        return

    def add_users(self, num=1):
        self.lmf._add_users(num=num)

    def train(self, users: Optional[Iterable[int]]=None):
        assert self.mode is 'train', "must be in train mode!"

        Utrain, Utest = self.data.make_training_datasets(dtype='sparse')
        U = Utrain + Utest

        AUCC = AUCCallback(
            self.data, np.arange(0,self.data.N,self.data.N//300,dtype=int),
            save_fn=lambda: self.lmf.save_as_epoch('best')
        )
        AUCC.set_model(self)
        LMFC = LMFCallback(Utrain, Utest, U, self.config['n_users'], self.config['n_items'])
        LMFC.set_model(self.lmf)
        trace = self.lmf.fit(Utrain, Utest, U, cb=[LMFC, AUCC], users=users)
        AUCC.save_result(os.path.join(self.model_file, 'AUC.csv'))
        LMFC.save_result(os.path.join(self.model_file, 'LMFC.csv'))
        return trace

    def train_update(self, users: Optional[Iterable[int]]=None):
        if users is None:
            users = np.arange(self.data.N)

        # TODO: improve so not required to construct the entire dataset for an update
        Utrain, Utest = self.data.make_training_datasets(dtype='sparse')
        U = Utrain + Utest

        AUCC = AUCCallback(self.data, users, save_fn=lambda: self.lmf.save_as_epoch('best_updated'), batchsize=100)
        AUCC.set_model(self)
        LMFC = LMFCallback(Utrain, Utest, U, self.config['n_users'], self.config['n_items'], users=users)
        LMFC.set_model(self.lmf)
        trace = self.lmf.fit(Utrain, Utest, U, users=users, cb=[LMFC, AUCC], exclude_phase={'Y'})
        AUCC.save_result(os.path.join(self.model_file, 'AUC_update.csv'))
        LMFC.save_result(os.path.join(self.model_file, 'LMFC_update.csv'))

        return trace,  AUCC.AUCe

    def save(self, path):
        self.lmf.save(os.path.join(path, 'model.h5'))
        self._save_config(path)

    def _save_config(self, lsh_path):
        with open(os.path.join(lsh_path, 'config.json'), 'w', encoding='utf-8') as fp:
            json.dump(self.config, fp, indent=4)

    def predict(self, u_in, i_in):
        # assert self.mode is 'predict'
        ret = self.lmf.predict(u_in, i_in)
        return {'phat': ret[0], 'q': ret[3]}


    def recommend(self, u_in: np.array):
        assert len(u_in.shape) == 1, "u_in must be one-dimensional array"
        nu = u_in.shape[0]
        ni = self.config['n_items']
        phats = self.predict(
            np.repeat(np.expand_dims(u_in, 0), ni).transpose().flatten(),
            np.tile(np.arange(ni), nu)
        )['phat']
        phats = phats.reshape(nu,-1)

        # uses ALOT of memory
        # idx, phats = tf_sort_scores(tf.constant(phats), axis=-1)
        # return np.array(idx), np.array(phats)
        return np.argsort(phats)[:, ::-1], np.sort(phats)[:, ::-1]

    def similar_to(self, i_in):
        nu = self.config['n_users']
        ni = self.config['n_items']
        Y = self.predict(np.zeros(shape=(nu,), dtype=int), np.arange(shape=(ni,), dtype=int))['q']
        y = Y[i_in]
        s = np.squeeze(cosine_similarity(Y, np.expand_dims(y, axis=0)))

        return np.argsort(s)[::-1][1:], np.sort(s)[::-1][1:]