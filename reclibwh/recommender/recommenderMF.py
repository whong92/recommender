import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from reclibwh.core.MatrixFactor import MatrixFactorizer
from reclibwh.core.AsymSVD import AsymSVD, AsymSVDCached
from reclibwh.utils.ItemMetadata import ExplicitDataFromCSV
from .recommenderInterface import Recommender
from ..utils.utils import get_pos_ratings_padded
from typing import Optional, Iterable
import tensorflow as tf
from reclibwh.utils.eval_utils import AUCCallback

class RecommenderMF(Recommender):

    def __init__(self, mode='train'
                 # arguments for train mode
                 , n_users=None, n_items=None, n_ranked=None
                 , mf_kwargs=None
                 # arguments for predict mode
                 , model_path=None, saved_model=None):

        super(RecommenderMF, self).__init__(model_path)
        
        self.mode = mode
        self.estimator = None
        self.config = {'n_users': n_users, 'n_items': n_items, 'n_ranked': n_ranked, 'mf_kwargs': mf_kwargs}
        if mode is 'train':
            if mf_kwargs is None:
                mf_kwargs = {'config_path': None}
            self.estimator = self._get_model()(model_path, n_users, n_items, n_ranked, **mf_kwargs)
        
        if mode is 'predict':

            assert model_path is not None
            config_path = os.path.join(model_path, 'config.json')

            with open(config_path, 'r', encoding='utf-8') as fp:
                self.config = json.load(fp)
            self.estimator = self._get_model()(
                model_path,
                self.config['n_users'], self.config['n_items'], self.config['n_ranked'], **self.config['mf_kwargs'], mode=mode,
                saved_model=saved_model if saved_model is not None else os.path.join(model_path, 'model_best.h5'),
            )
            self.data = None
            self.input_format = None

    def _get_model(self):
        return MatrixFactorizer

    def input_data(self, data: ExplicitDataFromCSV):
        self.data = data


    def train(self, early_stopping=True, tensorboard=True):
        assert self.mode is 'train', "must be in train mode!"
        self.estimator.fit(self.data, early_stopping=early_stopping, tensorboard=tensorboard)

    def save(self, model_path='model.h5'):
        self.estimator.save(model_path=model_path)
        self._save_config()

    def _save_config(self):
        with open(os.path.join(self.model_file, 'config.json'), 'w', encoding='utf-8') as fp:
            json.dump(self.config, fp, indent=4)

    def predict(self, u_in, i_in):
        bla = self.estimator.predict(np.array(u_in), np.array(i_in))
        bla = list(map(np.squeeze, bla))
        return {k:v for k,v in zip(['rhat', 'p', 'q'], bla)}

    def recommend(self, u_in):
        u_in = np.array(u_in)
        # manual prediction by enumerating all stuff
        nu = u_in.shape[0]
        ni = self.config['n_items']
        rhats = self.predict(
            u_in=np.repeat(np.expand_dims(u_in, 0), ni).transpose().flatten(),
            i_in=np.tile(np.arange(ni), nu)
        )['rhat']
        rhats = rhats.reshape(nu,-1)

        return np.argsort(rhats)[:, ::-1], np.sort(rhats)[:, ::-1]

    def train_update(self, users: Optional[Iterable[int]]=None, **kwargs):
        raise NotImplementedError

    def add_users(self, num=1):
        raise NotImplementedError

    def similar_to(self, i_in):
        p = self.predict(
            u_in=np.tile(np.array([0], dtype=np.int32), self.config['n_items']),
            i_in=np.arange(0, self.config['n_items'], dtype=np.int32))
        q_in = p['q'][i_in]
        q = p['q']
        s = np.squeeze(cosine_similarity(q, np.expand_dims(q_in, axis=0)))

        return np.argsort(s)[::-1][1:], np.sort(s)[::-1][1:]
    
    def evaluate(self):
        self.estimator.evaluate(self.data)
    

class RecommenderMFAsym(RecommenderMF):

    def __init__(self, *args, **kwargs):
        super(RecommenderMFAsym, self).__init__(*args, **kwargs)
        self.Bi = None

    def _get_model(self):
        return AsymSVD

    def predict(self, u_in, i_in):
        u_in = np.array(u_in)
        i_in = np.array(i_in)
        Utrain, U_test = self.data.make_training_datasets(users=np.unique(u_in), dtype='sparse')
        _, ys = get_pos_ratings_padded(Utrain, u_in, padding_val=0, offset_yp=1)
        bla = self.estimator.predict(u_in, i_in, ys)
        bla = list(map(np.squeeze, bla))
        return {k:v for k,v in zip(['rhat', 'p', 'q'], bla)}
    
    # AsymSVD doesn't really have a notion of users, so these are fine
    def train_update(self, users: Optional[Iterable[int]]=None, **kwargs):
        pass

    def add_users(self, num=1):
        pass

class RecommenderMFAsymCached(RecommenderMF):

    def __init__(self, mode='predict', model_path=None, saved_model=None, **kwargs):
        assert mode=='predict'
        saved_model = saved_model if saved_model is not None else \
            [os.path.join(model_path, 'model_best_X.h5'), os.path.join(model_path, 'model_best_main.h5')] 
        super(RecommenderMFAsymCached, self).__init__(mode=mode, model_path=model_path, saved_model=saved_model, **kwargs)
        self.Bi = None

    def _get_model(self):
        return AsymSVDCached
    
    def import_asym(self, rmfa: RecommenderMFAsym):
        Utrain, _ = self.data.make_training_datasets(users=None, dtype='sparse')
        if self.Bi is None: 
            self.Bi = np.array(self.data.get_item_mean_ratings(None)['rating_item_mean'])
        self.estimator.import_ASVD_weights(rmfa.estimator, Utrain, self.Bi)
    
    # AsymSVD doesn't really have a notion of users, so these are fine
    def train_update(self, users: Optional[Iterable[int]]=None, test:bool=False, **kwargs):
        
        Utrain, Utest = self.data.make_training_datasets(dtype='sparse', users=users)
        U = Utrain + Utest
        if self.Bi is None: 
            self.Bi = np.array(self.data.get_item_mean_ratings(None)['rating_item_mean'])
        auc_path = os.path.join(self.model_file, 'AUC_update.csv')
        AUCC = None
        if test:
            AUCC = AUCCallback(self.data, auc_path, users, save_fn=lambda: self.estimator.save('best_updated.h5'), batchsize=100)
            AUCC.set_model(self)
            AUCC.on_epoch_end(-1)
        self.estimator.update_users(U, self.Bi, users)
        if test:
            AUCC.on_epoch_end(0)
            AUCC.save_result(auc_path)
            self.estimator.evaluate(self.data, users=users)

        return AUCC.AUCe if AUCC is not None else None

    def add_users(self, num=1):
        self.config['n_users'] += num
        self.estimator._add_users(num=num)