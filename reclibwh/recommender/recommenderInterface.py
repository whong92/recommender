from reclibwh.utils.ItemMetadata import ExplicitData
from typing import Iterable, Optional
import numpy as np

class Recommender:
    def __init__(self, model_file=None):
        self.model_file = model_file
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        return

    def input_data(self, data: ExplicitData):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def train_update(self, users: Optional[Iterable[int]]=None, **kwargs):
        raise NotImplementedError

    def add_users(self, num=1):
        raise NotImplementedError

    def predict(self, items, users):
        raise NotImplementedError

    def recommend(self, user):
        raise NotImplementedError

    def similar_to(self, item, avg_items=True):
        raise NotImplementedError

    def recommend_similar_to(self, u_in, i_in, lamb=0.5):
        assert lamb >= 0. and lamb <= 1., "lamb not normalized"
        p = np.zeros(shape=(len(u_in), self.data.M), dtype=float)
        rec_aps, rec_ps = self.recommend(u_in)
        sim_aps, sim_ps = self.similar_to(i_in, avg_items=True)
        for u, (rec_ap, rec_p) in enumerate(zip(rec_aps, rec_ps)):
            p[u] = rec_p[np.argsort(rec_ap)]*(1.-lamb) + lamb*sim_ps[0][np.argsort(sim_aps[0])]
            print(rec_p[np.argsort(rec_ap)][0:10], sim_ps[0][np.argsort(sim_aps[0])][0:10], p[u][0:10])
        return np.argsort(p)[:, ::-1], np.sort(p)[:, ::-1]