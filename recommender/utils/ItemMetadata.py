from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import os
from .utils import csv2df, normalizeDf

class ExplicitData(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fetch_md(self, item_id):
        pass

    @abstractmethod
    def get_cat_map_user(self):
        pass

    @abstractmethod
    def get_cat_map_item(self):
        pass

    @abstractmethod
    def get_ratings(self):
        pass


class ExplicitDataFromCSV(ExplicitData):

    def __init__(self, ratings_csv, r_item_col, r_user_col, r_rating_col, metadata_csv, m_item_col, ):
        super(ExplicitDataFromCSV, self).__init__()
        self.ratings, self.user_map, self.item_map, self.N, self.M = \
            ExplicitDataFromCSV.sanitize_ratings(ratings_csv, r_item_col, r_user_col, r_rating_col)
        self.md_df = ExplicitDataFromCSV.makeMetadataDf(self.item_map.copy(), metadata_csv, m_item_col)
        return

    def fetch_md(self, item_ids):
        return self.md_df.loc[item_ids]

    def get_rating(self):
        return self.ratings

    def get_cat_map_user(self):
        return self.user_map

    def get_cat_map_item(self):
        return self.item_map

    @staticmethod
    def sanitize_ratings(ratings_csv, item, user, rating):
        return csv2df(ratings_csv, item, user, rating)

    @staticmethod
    def calc_rating_stats(df):
        rating_stats = pd.DataFrame({'r_count': df.groupby('item').count()['item', 'user']})
        rating_stats.set_index('item', inplace=True)
        return rating_stats

    @staticmethod
    def makeMetadataDf(item_map, item_md_csv, item_id_col):
        item_md_df = pd.read_csv(item_md_csv, index_col=item_id_col)
        item_map = item_map.merge(item_md_df, left_index=True, right_index=True)
        item_map.set_index('item_cat', inplace=True)
        item_map.drop('idx', inplace=True, axis=1)
        item_map.rename({'item_cat':'item'}, axis=1)
        return item_map

    def save(self, dir):
        self.ratings.to_csv(os.path.join(dir, 'ratings_sanitized.csv'))
        self.user_map.to_csv(os.path.join(dir, 'user_map.csv'))