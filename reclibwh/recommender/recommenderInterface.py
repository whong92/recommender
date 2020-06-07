from reclibwh.utils.ItemMetadata import ExplicitData
from typing import Iterable, Optional

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
