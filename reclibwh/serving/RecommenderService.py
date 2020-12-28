from reclibwh.core.Models import STANDARD_KERAS_SAVE_FMT, initialize_from_json
from reclibwh.core.ALS import ALSEnv
from reclibwh.core.AsymSVD import AsymSVDCachedEnv
from reclibwh.data.PresetIterators import ALS_data_iter_preset, MFAsym_data_iter_preset
from reclibwh.core.RecAlgos import RecAlgo
from reclibwh.core.Ensemble import EnsembleEnv
import numpy as np
import pandas as pd
import scipy.sparse as sps

from abc import ABC, abstractmethod
from reclibwh.utils.ItemMetadata import ExplicitData
from typing import Optional
import os, json

def filter_recommendations(recs, update_req, lim=200):

    rec_filt = {}
    for user, rec in recs.items():
        rated_items = update_req[user]['items']
        rec_items = np.array(rec['rec'])[:lim+len(rated_items)]
        rec_scores = np.array(rec['dist'])[:lim+len(rated_items)]
        items_idx = ~np.in1d(rec_items, rated_items)
        rec_filt[int(user)] = {
            'rec': rec_items[items_idx][:lim].tolist(),
            'dist': rec_scores[items_idx][:lim].tolist()
        }

    return rec_filt

def sanitize_update_req_data(update_req):
    return {int(k): v for k, v in update_req.items()}

def get_update_request_data(update_req: dict):
    """
    :param update_req: a dict of user: {items, ratings}
    :return:
    """

    rows = []
    cols = []
    vals = []

    for user, data in update_req.items():

        items = np.array(data['items'])
        ratings = np.array(data['ratings'])
        assert len(items) == len(ratings)

        rows.append(user*np.ones(shape=(len(items)), dtype=int))
        cols.append(items)
        vals.append(ratings)

    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def format_recommendation(users: np.array, items: np.array, scores: np.array):
    return {
        users[i]: {'rec': items[i].tolist(), 'dist': scores[i].tolist()} for i in range(len(users))
    }

def get_recommend_request_data(update_req: dict):
    """
    :param recommend_req: a list of users to recommend
    :return:
    """

    return list([k for k in update_req.keys()])

# some simple factory methods
def make_asymsvd_from_config(save_path, d: ExplicitData):

    item_mean_ratings = np.array(d.get_item_mean_ratings(None))
    Bi = np.array(item_mean_ratings)

    data_conf = {"M": d.M, "N": d.N}
    mc = initialize_from_json(data_conf=data_conf, config_path="SVD_asym_cached.json.template")

    env_varsc = {
        'data_conf': data_conf,
        'data': {'rnorm': {'loc': 0.0, 'scale': 5.0}, 'Bi': Bi}
    }
    return AsymSVDCachedEnv(save_path, mc, None, env_varsc)

def make_als_from_config(save_path, d: ExplicitData):

    data_conf = {"M": d.M, "N": d.N}
    env_vars = {
        "save_fmt": STANDARD_KERAS_SAVE_FMT,
        "data_conf": data_conf, "data": {}
    }
    m = initialize_from_json(data_conf=data_conf, config_path="ALS.json.template")
    return ALSEnv(save_path, m, d, env_vars)

def make_ensemble_from_configs(configs, data: ExplicitData):
    envs = []
    for config in configs:
        env_class = config['env_class']
        if env_class == 'AsymSVDCachedEnv': envs.append(make_asymsvd_from_config(config['save_path'], data))
        if env_class == 'ALSEnv': envs.append(make_als_from_config(config['save_path'], data))

    return envs

class BasicRecommenderService(ABC):

    def __init__(self, env: Optional[RecAlgo] = None):
        self.env = env

    @abstractmethod
    def user_update(self, update_req: dict):
        pass

    @abstractmethod
    def item_similar_to(self, *args, **kwargs):
        pass

    def user_recommend(self, users: list):
        users = np.array(users)
        items, scores = self.env.recommend(users)
        return format_recommendation(users, items, scores)

    def update_and_recommend(self, update_req):
        self.user_update(update_req)
        recs = self.user_recommend(get_recommend_request_data(update_req))
        return filter_recommendations(recs, update_req)


class ALSRecommenderService(BasicRecommenderService):

    def __init__(self, save_path, data: ExplicitData):
        super(ALSRecommenderService, self).__init__(None)
        self.save_path = save_path
        self.env = make_als_from_config(save_path, data)
        self.env.predict(np.array([0]), np.array([0])) # to make sure environment initialized

    def user_update(self, update_req: dict):

        rows, cols, vals = get_update_request_data(update_req)
        update_data = self.env.make_update_data((rows, cols, vals))
        self.env.update_user(update_data)

    def item_similar_to(self, *args, **kwargs):

        # old implementation
        # i_in = np.array(i_in)
        # y = self.als.Y[i_in]
        # Y = self.als.Y
        # s = cosine_similarity(y, Y)
        # if avg_items: s = np.mean(s, axis=0, keepdims=True)
        #
        # return np.argsort(s)[:, ::-1], np.sort(s)[:, ::-1]

        raise NotImplementedError


class MFAsymRecService(BasicRecommenderService):

    def __init__(self, save_path, data: ExplicitData):

        super(MFAsymRecService, self).__init__(None)
        self.save_path = save_path
        self.env = make_asymsvd_from_config(save_path, data)
        self.env.predict(np.array([0]), np.array([0]))  # to make sure environment initialized

    def user_update(self, update_req: dict):

        rows, cols, vals = get_update_request_data(update_req)
        update_data = self.env.make_update_data((rows, cols, vals))
        self.env.update_user(update_data)

    def item_similar_to(self, *args, **kwargs):

        # old implementation
        # i_in = np.array(i_in)
        # p = self.predict(
        #     u_in=np.tile(np.array([0], dtype=np.int32), self.config['n_items']),
        #     i_in=np.arange(0, self.config['n_items'], dtype=np.int32))
        # q_in = p['p'][i_in]
        # q = p['p']
        # s = cosine_similarity(q_in, q)
        # if avg_items: s = np.mean(s, axis=0, keepdims=True)
        #
        # return np.argsort(s)[:, ::-1], np.sort(s)[:, ::-1]

        raise NotImplementedError

class ALSMFEnsembleService(BasicRecommenderService):

    def __init__(self, save_path, data: ExplicitData):

        super(ALSMFEnsembleService, self).__init__(None)

        with open(os.path.join(save_path, "config.json")) as fp: config = json.load(fp)
        envs = make_ensemble_from_configs(config, data)
        self.env = EnsembleEnv(None, envs, None, {'data_conf': {"M": data.M, "N": data.N}})

    def user_update(self, update_req: dict):

        rows, cols, vals = get_update_request_data(update_req)
        update_data = self.env.make_update_data((rows, cols, vals))
        self.env.update_user(update_data)

    def item_similar_to(self, *args, **kwargs):

        raise NotImplementedError
