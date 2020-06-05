import json
import os

import numpy as np
import scipy.sparse as sps
from sklearn.metrics.pairwise import cosine_similarity

from reclibwh.utils.ItemMetadata import ExplicitData
from .recommenderInterface import Recommender
from ..utils.eval_utils import AUCCallback
from typing import Iterable, Optional, Union

class RecommenderEnsemble(Recommender):

    def __init__(self, recommenders):

        super(RecommenderEnsemble, self).__init__(None)
        self.data = None
        self.recommenders = recommenders
        self.config = self.recommenders[0].config

    def input_data(self, data: Union[Iterable[ExplicitData], ExplicitData]):
        self.data = data
        if type(data) is ExplicitData:
            self.data = [data]
            for rec in self.recommenders: rec.input_data(data)
        else:
            for rec, d in zip(self.recommenders, data): rec.input_data(d)
        return

    def add_users(self, num=1):
        for rec in self.recommenders: rec.add_users(num=num)

    def train(self):
        # ensemble model only accepts pre-trained recommenders
        raise NotImplementedError

    def train_update(self, users: Optional[Iterable[int]]=None):
        for rec in self.recommenders: rec.train_update(users)

    def save(self, path):
        for rec in self.recommenders: rec.save(path)

    def _save_config(self, lsh_path):
        return

    def predict(self, u_in, i_in):
        rhats = np.ones(shape=(len(u_in)), dtype=float)
        for rec in self.recommenders: rhats = np.multiply(rhats, rec.predict(u_in, i_in))

    def recommend(self, u_in):
        p = np.ones(shape=(len(u_in), self.data[0].M), dtype=float)
        for rec in self.recommenders:
            rec_aps, rec_ps = rec.recommend(u_in)
            for u, (rec_ap, rec_p) in enumerate(zip(rec_aps, rec_ps)):
                p[u] = np.multiply(p[u], rec_p[np.argsort(rec_ap)])
        return np.argsort(p)[:, ::-1], np.sort(p)[:, ::-1]

    def similar_to(self, i_in):
        p = np.ones(shape=(len(i_in), self.data[0].M), dtype=float)
        for rec in self.recommenders:
            rec_aps, rec_ps = rec.similar_to(i_in)
            for u, (rec_ap, rec_p) in enumerate(zip(rec_aps, rec_ps)):
                p[u] = np.multiply(p[u], rec_p[np.argsort(rec_ap)])
        return np.argsort(p)[:, ::-1], np.sort(p)[:, ::-1]

    def recommend_similar_to(self, u_in, i_in, lamb=0.5):
        p = np.ones(shape=(len(i_in), self.data[0].M), dtype=float)
        for rec in self.recommenders:
            rec_aps, rec_ps = rec.recommend_similar_to(u_in, i_in, lamb=lamb)
            for u, (rec_ap, rec_p) in enumerate(zip(rec_aps, rec_ps)):
                p[u] = np.multiply(p[u], rec_p[np.argsort(rec_ap)])
        return np.argsort(p)[:, ::-1], np.sort(p)[:, ::-1]