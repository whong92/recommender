from abc import ABC, abstractmethod
import pandas as pd
import os
from .utils import csv2df, normalizeDf, splitDf
import tensorflow as tf

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

    def from_raw_csv(self, ratings_csv, r_item_col, r_user_col, r_rating_col, metadata_csv, m_item_col, ):
        self.ratings, self.user_map, self.item_map = \
            ExplicitDataFromCSV.sanitize_ratings(ratings_csv, r_item_col, r_user_col, r_rating_col)
        self.md_df = ExplicitDataFromCSV.makeMetadataDf(self.item_map.copy(), metadata_csv, m_item_col)
        self.stats = ExplicitDataFromCSV.calc_rating_stats(self.ratings)

    def from_saved_csv(self, ratings_csv, user_map_csv, item_map_csv, md_csv, stats_csv):
        self.ratings = pd.read_csv(ratings_csv)
        self.user_map = pd.read_csv(user_map_csv, index_col='user_cat')
        self.item_map = pd.read_csv(item_map_csv, index_col='item_cat')
        self.md_df = pd.read_csv(md_csv, index_col='item_cat')
        self.stats = pd.read_csv(stats_csv, index_col='item')

    def __init__(self, from_saved=False, **kwargs):
        super(ExplicitDataFromCSV, self).__init__()
        if from_saved:
            self.from_saved_csv(**kwargs)
        else:
            self.from_raw_csv(**kwargs)
        self.N, self.M = len(self.user_map), len(self.item_map)
        return

    def fetch_md(self, item_ids):
        return self.md_df.loc[item_ids]

    def get_ratings(self):
        return self.ratings

    def get_cat_map_user(self):
        return self.user_map

    def get_cat_map_item(self):
        return self.item_map

    @staticmethod
    def sanitize_ratings(ratings_csv, item, user, rating):
        return normalizeDf(csv2df(ratings_csv, item, user, rating))

    @staticmethod
    def calc_rating_stats(df):
        rating_stats = df.groupby('item', as_index=False).count()[['item', 'user']]
        rating_stats.set_index('item', inplace=True)
        rating_stats.rename({'user': 'r_count'}, inplace=True)
        return rating_stats

    @staticmethod
    def makeMetadataDf(item_map, item_md_csv, item_id_col):
        item_md_df = pd.read_csv(item_md_csv, index_col=item_id_col)
        item_map = item_map.merge(item_md_df, left_index=True, right_index=True)
        item_map.set_index('item_cat', inplace=True)
        item_map.rename({'item_cat':'item'}, axis=1, inplace=True)
        return item_map

    def save(self, dir):
        self.ratings.to_csv(os.path.join(dir, 'ratings_sanitized.csv'))
        self.user_map.to_csv(os.path.join(dir, 'user_map.csv'))
        self.item_map.to_csv(os.path.join(dir, 'item_map.csv'))
        self.md_df.to_csv(os.path.join(dir, 'metadata.csv'))
        self.stats.to_csv(os.path.join(dir, 'stats.csv'))

    def make_training_datasets(self, train_test_split=0.8):
        D_train, D_test = splitDf(self.ratings, train_test_split)
        feat_train = {'user': D_train['user'], 'item': D_train['item'], 'rating': D_train['rating']}
        feat_test = {'user': D_test['user'], 'item': D_test['item'], 'rating': D_test['rating']}
        ds_train_input_fn = lambda: tf.data.Dataset.from_tensor_slices(feat_train)
        ds_test_input_fn = lambda: tf.data.Dataset.from_tensor_slices(feat_test)
        return ds_train_input_fn, ds_test_input_fn, D_train, D_test


if __name__=="__main__":
    """
    ratings_csv = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\ratings.csv'
    movies_csv = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\movies.csv'
    d = ExplicitDataFromCSV(False,
                            ratings_csv=ratings_csv, r_item_col='movieId', r_user_col='userId',
                            r_rating_col='rating',
                            metadata_csv=movies_csv, m_item_col='movieId')
    d.save('D:\\PycharmProjects\\recommender\\data\\ml-latest-small')
    """
    ratings_csv = 'D:\\PycharmProjects\\recommender\\data\\ml-20m\\ratings_sanitized.csv'
    user_map_csv = 'D:\\PycharmProjects\\recommender\\data\\ml-20m\\user_map.csv'
    item_map_csv = 'D:\\PycharmProjects\\recommender\\data\\ml-20m\\item_map.csv'
    md_csv = 'D:\\PycharmProjects\\recommender\\data\\ml-20m\\metadata.csv'
    stats_csv = 'D:\\PycharmProjects\\recommender\\data\\ml-20m\\stats.csv'
    d = ExplicitDataFromCSV(True,
                            ratings_csv=ratings_csv,
                            user_map_csv=user_map_csv,
                            item_map_csv=item_map_csv,
                            md_csv=md_csv,
                            stats_csv=stats_csv)
    print(d.ratings.head())