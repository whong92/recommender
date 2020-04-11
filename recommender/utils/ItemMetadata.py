from abc import ABC, abstractmethod
import pandas as pd
import os
from .utils import csv2df, normalizeDf, splitDf
import tensorflow as tf
import numpy as np
import scipy.sparse as sps

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
        self.df_train = None
        self.df_test = None

    def from_saved_csv(self, ratings_csv, user_map_csv, item_map_csv, md_csv, stats_csv, ratings_train_csv=None, ratings_test_csv=None):
        self.ratings = pd.read_csv(ratings_csv, index_col=None)[['rating', 'item', 'user']]
        self.user_map = pd.read_csv(user_map_csv, index_col='user_cat')
        self.item_map = pd.read_csv(item_map_csv, index_col='item_cat')
        self.md_df = pd.read_csv(md_csv, index_col='item_cat')
        self.stats = pd.read_csv(stats_csv, index_col='item')
        self.df_train = pd.read_csv(ratings_train_csv, index_col=None)[['rating', 'item', 'user']]
        self.df_test = pd.read_csv(ratings_test_csv, index_col=None)[['rating', 'item', 'user']]
        self.df_test.set_index('user', drop=False, inplace=True)
        self.df_train.set_index('user', drop=False, inplace=True)

    def from_standard_dataset(self, data_folder):
        self.from_saved_csv(
            ratings_csv=os.path.join(data_folder, 'ratings_sanitized.csv'),
            user_map_csv = os.path.join(data_folder, 'user_map.csv'),
            item_map_csv = os.path.join(data_folder, 'item_map.csv'),
            md_csv = os.path.join(data_folder, 'metadata.csv'),
            stats_csv = os.path.join(data_folder, 'stats.csv'),
            ratings_train_csv = os.path.join(data_folder, 'ratings_train.csv'),
            ratings_test_csv = os.path.join(data_folder, 'ratings_test.csv'),
        )

    def __init__(self, from_saved=False, **kwargs):
        """
        The user/item columns in all stored tables  in this class are in terms of the
        sanitized user/items (0-M/0-N) respectively. self.item_map and self.user_map
        provides the mapping back to its original keys
        :param from_saved: if we are loading from saved (processed) data, in which case a
        folder with the required csvs is expected
        :param kwargs: arguments that are required by ExplicitDataFromCSV.from_raw_csv
        """
        super(ExplicitDataFromCSV, self).__init__()
        if from_saved:
            self.from_standard_dataset(**kwargs)
        else:
            self.from_raw_csv(**kwargs)
        self.df_train_by_user = None
        self.df_test_by_user = None
        return

    def fetch_md(self, item_ids):
        return self.md_df.loc[item_ids]

    def get_ratings(self):
        return self.ratings

    def get_cat_map_user(self):
        return self.user_map

    def get_cat_map_item(self):
        return self.item_map

    @property
    def N(self):
        return len(self.user_map)

    @property
    def M(self):
        return len(self.item_map)

    def add_user(self):
        self.user_map = self.user_map.append(pd.DataFrame({'user': ['manual_add_{:d}'.format(self.N)]}, index=[self.N]))

    def add_user_ratings(self, users, items, ratings, train=True):
        for u in users: assert u in self.user_map.index
        update = pd.DataFrame({'user': users, 'item': items, 'rating': ratings}).set_index('user', drop=False)
        self.ratings = self.ratings.append(update)
        if train: self.df_train = self.df_train.append(update)
        else: self.df_test = self.df_test.append(update)
        self.update_stats(update)
        return

    def pop_user_ratings(self, users, train=True):
        for u in users: assert u in self.user_map.index
        # TODO: fix, this is wrong!
        self.ratings.drop(index=users, inplace=True, errors='ignore')
        if train:
            users = list(filter(lambda x:x in self.df_train.index, users))
            dropped = self.df_train.loc[users,:]
            self.df_train.drop(index=users, inplace=True, errors='ignore') # teehee
        else:
            users = list(filter(lambda x: x in self.df_test.index, users))
            dropped = self.df_test.loc[users,:]
            self.df_test.drop(index=users, inplace=True, errors='ignore')
        self.update_stats(dropped, False)
        return dropped

    @staticmethod
    def sanitize_ratings(ratings_csv, item, user, rating):
        return normalizeDf(csv2df(ratings_csv, item, user, rating))

    @staticmethod
    def calc_rating_stats(df):
        rating_stats = df.groupby('item', as_index=False).count()[['item', 'user']]
        rating_stats.set_index('item', inplace=True)
        rating_stats.rename({'user': 'r_count'}, inplace=True)
        return rating_stats

    def update_stats(self, update: pd.DataFrame, add=True):
        # TODO: fix
        for _, r in update.iterrows():
            if add: self.stats.loc[r['item'], 'r_count'] += 1
            else: self.stats.loc[r['item'], 'r_count'] -= 1
        return

    @staticmethod
    def makeMetadataDf(item_map, item_md_csv, item_id_col):
        item_md_df = pd.read_csv(item_md_csv, index_col=item_id_col)
        item_map = item_map.merge(item_md_df, left_index=True, right_index=True)
        item_map.set_index('item_cat', inplace=True)
        item_map.rename({'item_cat':'item'}, axis=1, inplace=True)
        return item_map

    def save(self, dir):
        self.ratings.to_csv(os.path.join(dir, 'ratings_sanitized.csv'), index=False)
        self.user_map.to_csv(os.path.join(dir, 'user_map.csv'))
        self.item_map.to_csv(os.path.join(dir, 'item_map.csv'))
        self.md_df.to_csv(os.path.join(dir, 'metadata.csv'))
        self.stats.to_csv(os.path.join(dir, 'stats.csv'))
        if self.df_train is not None and self.df_test is not None:
            self.df_train.to_csv(os.path.join(dir, 'ratings_train.csv'), index=False)
            self.df_test.to_csv(os.path.join(dir, 'ratings_test.csv'), index=False)

    def make_training_datasets(
            self, train_test_split=0.8,
            dtype:str='dense', ctype:np.dtype=np.int32, rtype:np.dtype=np.float64):
        if self.df_train is None or self.df_test is None:
            D_train, D_test = splitDf(self.ratings, train_test_split)
        else:
            D_train, D_test = self.df_train, self.df_test
        assert dtype in {'dense', 'sparse'}, "dtype not understood"
        if dtype == 'dense':
            return self.make_positive_arrays(D_train, ctype, rtype), \
                   self.make_positive_arrays(D_test, ctype, rtype)
        elif dtype == 'sparse':
            return self.make_sparse_matrices(D_train, ctype, rtype), \
                   self.make_sparse_matrices(D_test, ctype, rtype)

    def make_positive_arrays(self, D:pd.DataFrame, ctype:np.dtype=np.int32, rtype:np.dtype=np.float64):
        return (
            np.array(D['user'], dtype=ctype),
            np.array(D['item'], dtype=ctype),
            np.array(D['rating'], dtype=rtype)
        )

    def make_sparse_matrices(self, D:pd.DataFrame, ctype:np.dtype=np.int32, rtype:np.dtype=np.float64):
        rows, cols, data = self.make_positive_arrays(D, ctype, rtype)
        return sps.csr_matrix((data, (cols, rows)), shape=(self.M, self.N))

    def get_user_ratings(self, user, train=False):
        ratings = pd.DataFrame(columns=self.df_train.keys())
        if train and user in self.df_train.index:
            ratings = self.df_train.loc[[user]]
        if not train and user in self.df_test.index:
            ratings = self.df_test.loc[[user]]
        return ratings

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