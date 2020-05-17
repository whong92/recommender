from abc import ABC, abstractmethod
import pandas as pd
import os
from .utils import csv2df, normalizeDf, splitDf
import numpy as np
import scipy.sparse as sps
from typing import Iterable, Optional
import sqlite3
from collections import defaultdict

class ExplicitData(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fetch_md(self, item_id):
        pass

    @abstractmethod
    def get_ratings(self):
        pass

    @abstractmethod
    def add_user(self):
        pass

    @abstractmethod
    def add_user_ratings(self, users, items, ratings, train=True):
        pass

    @abstractmethod
    def pop_user_ratings(self, users, train=True):
        pass

    @abstractmethod
    def make_training_datasets(self, dtype:str='dense', ctype:np.dtype=np.int32, rtype:np.dtype=np.float64, users:Optional[Iterable[int]]=None):
        pass

    @property
    def N(self):
        pass

    @property
    def M(self):
        pass

class ExplicitDataFromSql3(ExplicitData):

    def __init__(self, sql3_f, rt, rt_user_col, rt_item_fk_col, rt_rating_col, it, it_item_id_col, ut,  ut_id_col, user_offset):
        self.sql3_f = sql3_f
        self.Noffset = user_offset

        self.conf = {
            'rating_table': rt,
            'rt_user_col': rt_user_col,
            'rt_item_fk_col': rt_item_fk_col,
            'rt_rating_col': rt_rating_col,
            'item_table': it,
            'it_item_id_col': it_item_id_col,
            'user_table': ut,
            'ut_id_col': ut_id_col
        }
    
    
    def get_conn(self):
        conn = sqlite3.connect(self.sql3_f)
        conn.row_factory = sqlite3.Row
        return conn
    
    @staticmethod
    def _sqlite_rows_2_dataframe(rows, columns):
        data = defaultdict(list)
        for row in rows:
            cols = rows[0].keys()
            for col,v in zip(cols, row): data[col].append(v)
        return pd.DataFrame(data, columns=columns)

    def fetch_md(self, item_id):
        # ain't nobody got time for this
        raise NotImplementedError

    def get_ratings(self):
        
        # TODO: add user rating to this
        cur = self.get_conn().cursor()
        
        rt  = self.conf['rating_table']
        it  = self.conf['item_table']
        rt_u  = self.conf['rt_user_col']
        rt_ifk  = self.conf['rt_item_fk_col']
        rt_r  = self.conf['rt_rating_col']
        it_i  = self.conf['it_item_id_col']

        cmd = 'SELECT {rt_ifk}, {rt_r} FROM {rt}'.format(rt_r=rt_r, rt=rt, rt_ifk=rt_ifk)
        cur.execute(
            cmd
        )
        res = cur.fetchall()
        return ExplicitDataFromSql3._sqlite_rows_2_dataframe(res, columns=[rt_ifk, rt_r])

    def add_user(self):
        raise NotImplementedError

    def add_user_ratings(self, users, items, ratings, train=True):

        conn = self.get_conn()
        cur = conn.cursor()

        rt  = self.conf['rating_table']
        it  = self.conf['item_table']
        rt_u  = self.conf['rt_user_col']
        rt_ifk  = self.conf['rt_item_fk_col']
        rt_r  = self.conf['rt_rating_col']

        insert_str = 'INSERT INTO {rt} ({rt_ifk},{rt_r}) VALUES (?,?) '\
            'ON CONFLICT({rt_ifk}) DO UPDATE SET {rt_r}=excluded.{rt_r}'\
            .format(rt=rt, rt_u=rt_u, rt_ifk=rt_ifk, rt_r=rt_r)
        cur.executemany(insert_str, [(u,i,r) for (u,i,r) in zip(users, items, ratings)])
        conn.commit()

    def pop_user_ratings(self, users, train=True):
        raise NotImplementedError

    def make_training_datasets(self, dtype:str='dense', ctype:np.dtype=np.int32, rtype:np.dtype=np.float64, users:Optional[Iterable[int]]=None):
        
        assert len(users) > 0
        if users is None: users = np.arange(self.N)

        conn = self.get_conn()

        cur = conn.cursor()
        
        rt  = self.conf['rating_table']
        it  = self.conf['item_table']
        rt_u  = self.conf['rt_user_col']
        rt_ifk  = self.conf['rt_item_fk_col']
        rt_r  = self.conf['rt_rating_col']
        it_i  = self.conf['it_item_id_col']

        cmd = 'SELECT {rt_ifk}, {rt_r}, {rt_u} FROM {rt}'.format(rt_r=rt_r, rt=rt, rt_ifk=rt_ifk, rt_u=rt_u)
        cond_str = ""

        users = [(user - self.Noffset) for user in users]
        if users is not None:
            cond_str = ['{rt_u}=?'.format(rt_u=rt_u)]*len(users)
            cond_str = "(" + " OR ".join(cond_str) + ")"
            cmd += ' WHERE ' + cond_str
            rows = cur.execute(cmd, tuple(users)).fetchall()
        else:
            rows = cur.execute(cmd).fetchall()
        
        conn.commit()
        
        D = ExplicitDataFromSql3._sqlite_rows_2_dataframe(
            rows,
            columns = [rt_u, rt_ifk, rt_r]
        )
        D.rename(
            columns={rt_u: 'user', rt_ifk: 'item', rt_r: 'rating'},
            inplace=True,
        )
        D['user'] += self.Noffset

        emptyD = pd.DataFrame({'user':[], 'item':[], 'rating':[]})
        
        if dtype=='dense':
            return ExplicitDataFromCSV.make_positive_arrays(D, ctype=ctype, rtype=rtype), ExplicitDataFromCSV.make_positive_arrays(emptyD, ctype=ctype, rtype=rtype)
        else:
            return ExplicitDataFromCSV.make_sparse_matrices(D, self.N, self.M, ctype=ctype, rtype=rtype), ExplicitDataFromCSV.make_sparse_matrices(emptyD, self.N, self.M, ctype=ctype, rtype=rtype)

    @property
    def M(self):
        conn = self.get_conn()
        cur = conn.cursor()
        it  = self.conf['item_table']        
        res = cur.execute('SELECT COUNT(*) FROM {it}'.format(it=it)).fetchall()
        conn.commit()
        return ExplicitDataFromSql3._sqlite_rows_2_dataframe(res, ['COUNT(*)']).iloc[0]['COUNT(*)']

    @property
    def N(self):
        conn = self.get_conn()
        cur = conn.cursor()
        ut  = self.conf['user_table']  
        ut_id_col = self.conf['ut_id_col']
        res = cur.execute('SELECT MAX({ut_id_col}) FROM {ut}'.format(ut_id_col=ut_id_col, ut=ut)).fetchall()
        conn.commit()
        colname = 'MAX({ut_id_col})'.format(ut_id_col=ut_id_col)
        return ExplicitDataFromSql3._sqlite_rows_2_dataframe(res, [colname]).iloc[0][colname] + 1 + self.Noffset
    
    def add_items(self, **kwargs):
        """
        Utility function for importing stuff from ExplicitDataFromCSV (or others) into ExplicitDataFromSql3
        """
        assert 'item_ids' in kwargs, "required: item_ids"
        assert 'names' in kwargs, "required: names"
        conn = self.get_conn()
        cur = conn.cursor()

        it  = self.conf['item_table']
        it_i  = self.conf['it_item_id_col']


        insert_str = 'INSERT INTO {it} ({it_i}, created_at, name, desc, poster) VALUES (?, ?, ?, ?, ?) '.format(it=it, it_i=it_i)

        cur.executemany(insert_str, [(idx, '2020-04-25 20:43:24.088110', name, desc, poster) for (idx, name, desc, poster) in zip(kwargs['item_ids'], kwargs['names'], kwargs['desc'], kwargs['poster_path'])])
        conn.commit()


class ExplicitDataFromCSV(ExplicitData):

    def from_raw_csv(self, ratings_csv, r_item_col, r_user_col, r_rating_col, metadata_csv, m_item_col, train_test_split=0.8):
        self.ratings, self.user_map, self.item_map = \
            ExplicitDataFromCSV.sanitize_ratings(ratings_csv, r_item_col, r_user_col, r_rating_col)
        _, test_split = splitDf(ratings_csv, train_test_split)
        self.ratings['split'] = 0
        self.ratings.iloc[test_split]['split'] = 1
        self.md_df = ExplicitDataFromCSV.makeMetadataDf(self.item_map.copy(), metadata_csv, m_item_col)

    def from_saved_csv(self, ratings_csv, split_csv, user_map_csv, item_map_csv, md_csv):
        self.ratings = pd.read_csv(ratings_csv, index_col=None)[['rating', 'item', 'user']]
        self.ratings_split  = pd.read_csv(split_csv, index_col=None)
        assert len(self.ratings) == len(self.ratings_split)
        self.ratings['split'] = self.ratings_split
        self.ratings.set_index(['user', 'split'], drop=False, inplace=True)
        self.user_map = pd.read_csv(user_map_csv, index_col='user_cat')
        self.item_map = pd.read_csv(item_map_csv, index_col='item_cat')
        self.md_df = pd.read_csv(md_csv, index_col='item_cat')

    def from_standard_dataset(self, data_folder):
        self.from_saved_csv(
            ratings_csv=os.path.join(data_folder, 'ratings_sanitized.csv'),
            split_csv=os.path.join(data_folder, 'ratings_split.csv'),
            user_map_csv = os.path.join(data_folder, 'user_map.csv'),
            item_map_csv = os.path.join(data_folder, 'item_map.csv'),
            md_csv = os.path.join(data_folder, 'metadata.csv'),
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
    def df_train(self):
        return self.get_ratings_split(0)
    
    @property
    def df_test(self):
        return self.get_ratings_split(1)

    @property
    def N(self):
        return len(self.user_map)

    @property
    def M(self):
        return len(self.item_map)
    
    @property
    def Nranked(self):
        return len(self.ratings)
    
    @property
    def stats(self):
        return ExplicitDataFromCSV.calc_rating_stats(self.ratings)
    
    def get_ratings_split(self, split):
        if split not in self.ratings.index: return pd.DataFrame({}, columns=self.ratings.columns)
        return self.ratings.loc[self.ratings.split==split]

    def add_user(self):
        self.user_map = self.user_map.append(
            pd.DataFrame({'user': ['manual_add_{:d}'.format(self.N)]}, index=[self.N])
        )

    def add_user_ratings(self, users, items, ratings, train=True):
        for u in users: assert u in self.user_map.index
        update = pd.DataFrame(
            {'user': users, 'item': items, 'rating': ratings, 'split': [int(not train)]*len(users)}
        ).set_index(['user', 'split'], drop=False)
        self.ratings = self.ratings.append(update)
        return update

    def pop_user_ratings(self, users, train=True):
        for u in users: assert u in self.user_map.index
        split = int(not train)
        dropped = self.ratings.loc[[(u, split) for u in users]]
        # need to pull out add the other split back, because drop doesn't let us drop based on two levels at once!
        addback = self.ratings.loc[[(u, 1-split) for u in users]]
        
        self.ratings.drop(index=users, level=0, inplace=True, errors='ignore')
        self.ratings = self.ratings.append(addback)
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

    @staticmethod
    def makeMetadataDf(item_map, item_md_csv, item_id_col):
        item_md_df = pd.read_csv(item_md_csv, index_col=item_id_col)
        item_map = item_map.merge(item_md_df, left_index=True, right_index=True)
        item_map.set_index('item_cat', inplace=True)
        item_map.rename({'item_cat':'item'}, axis=1, inplace=True)
        return item_map

    def save(self, dir):
        self.ratings[['rating', 'item', 'user']].to_csv(os.path.join(dir, 'ratings_sanitized.csv'), index=False)
        self.ratings[['split']].to_csv(os.path.join(dir, 'ratings_split.csv'), index=False)
        self.user_map.to_csv(os.path.join(dir, 'user_map.csv'))
        self.item_map.to_csv(os.path.join(dir, 'item_map.csv'))
        self.md_df.to_csv(os.path.join(dir, 'metadata.csv'))

    def make_training_datasets(self, dtype:str='dense', ctype:np.dtype=np.int32, rtype:np.dtype=np.float64, users:Optional[Iterable[int]]=None):
        if users is not None:
            D_train = self.get_user_ratings(users, train=True)
            D_test = self.get_user_ratings(users, train=False)
        else:
            D_train, D_test = self.df_train, self.df_test
        assert dtype in {'dense', 'sparse'}, "dtype not understood"
        if dtype == 'dense':
            return ExplicitDataFromCSV.make_positive_arrays(D_train, ctype, rtype), \
                   ExplicitDataFromCSV.make_positive_arrays(D_test, ctype, rtype)
        elif dtype == 'sparse':
            return ExplicitDataFromCSV.make_sparse_matrices(D_train, self.N, self.M, ctype, rtype), \
                   ExplicitDataFromCSV.make_sparse_matrices(D_test, self.N, self.M, ctype, rtype)

    @staticmethod
    def make_positive_arrays(D:pd.DataFrame, ctype:np.dtype=np.int32, rtype:np.dtype=np.float64):
        return (
            np.array(D['user'], dtype=ctype),
            np.array(D['item'], dtype=ctype),
            np.array(D['rating'], dtype=rtype)
        )

    @staticmethod
    def make_sparse_matrices(D:pd.DataFrame, N:int, M:int, ctype:np.dtype=np.int32, rtype:np.dtype=np.float64):
        rows, cols, data = ExplicitDataFromCSV.make_positive_arrays(D, ctype, rtype)
        return sps.csr_matrix((data, (rows, cols)), shape=(N,M))

    def get_user_ratings(self, users, train=False):
        split = int(not train)
        users = [(u,split) for u in users]
        users = self.ratings.index.intersection(users)
        ratings = self.ratings.loc[users]

        return ratings

if __name__=="__main__":
    """
    # example making new dataset
    ratings_csv = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\ratings.csv'
    movies_csv = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\movies.csv'
    d = ExplicitDataFromCSV(False,
                            ratings_csv=ratings_csv, r_item_col='movieId', r_user_col='userId',
                            r_rating_col='rating',
                            metadata_csv=movies_csv, m_item_col='movieId')
    d.save('D:\\PycharmProjects\\recommender\\data\\ml-latest-small')
    """
    
    """
    # convert old dataset format to new
    ratings_train_csv = "/home/ong/personal/recommender/data/ml-20m/ratings_train.csv"
    ratings_test_csv = "/home/ong/personal/recommender/data/ml-20m/ratings_test.csv"
    df_train = pd.read_csv(ratings_train_csv, index_col=None)
    df_test = pd.read_csv(ratings_test_csv, index_col=None)
    ratings_all = pd.concat([df_train, df_test])
    df_split = pd.DataFrame({
        'split' : np.concatenate([
            np.zeros(shape=len(df_train,), dtype=int), 
            np.ones(shape=len(df_test,), dtype=int)
        ])
    })
    ratings_all.reset_index(drop=True).to_csv('/home/ong/personal/recommender/data/ml-20m-2/ratings_sanitized.csv', index=False)
    df_split.to_csv('/home/ong/personal/recommender/data/ml-20m-2/ratings_split.csv', index=False)
    """

    # data_dir = "/home/ong/personal/recommender/data/ml-latest-small-2"
    # d = ExplicitDataFromCSV(True, data_folder=data_dir)
    # print(d.get_user_ratings([0]))
    # print(d.get_user_ratings([0], train=True))
    # Utrain, Utest = d.make_training_datasets(users=[0], dtype='sparse')
    # print(Utrain[0,:])
    # print(Utest[0,:])


    """
    d = ExplicitDataFromSql3(
        '/home/ong/personal/FiML/FiML/db.sqlite3',
        rt='backend_rating',
        rt_user_col='user_id',
        rt_item_fk_col='film_id',
        rt_rating_col='rating',
        it='backend_film', 
        it_item_id_col='id',
        ut='auth_user',
        ut_id_col='id',
        user_offset=610
    )

    # cur = d.conn.cursor()
    print(d.M)
    print(d.N)
    print(d.get_ratings())
    
    print(d.make_training_datasets(users=[611, 612, 613]))

    print(d.make_training_datasets(users=[617], dtype='sparse'))

    # d.add_user_ratings([1,2,3], [3,4,5], [5.0, 5.0, 5.0])
    """