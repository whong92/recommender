from abc import ABC, abstractmethod
import pandas as pd
import os
from .utils import csv2df, normalizeDf, splitDf
import numpy as np
import scipy.sparse as sps
from typing import Iterable, Optional
import sqlite3
from collections import defaultdict
import psycopg2
import psycopg2.extras

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
    def make_training_datasets(self, dtype:str='dense', ctype:np.dtype=np.int32, rtype:np.dtype=np.float64, users:Optional[Iterable[int]]=None):
        pass

    @property
    def N(self):
        pass

    @property
    def M(self):
        pass

    def get_item_mean_ratings(self, items):
        return

class ExplicitDataFromSql3(ExplicitData):

    # TODO: put all of this shit in a config
    def __init__(self, sql3_f, rt, rt_user_col, rt_item_fk_col, rt_rating_col, it, it_item_id_col, it_item_mean_col, ut,  ut_id_col, user_offset, normalize=None):
        self.sql3_f = sql3_f
        self.Noffset = user_offset

        self.conf = {
            'rating_table': rt,
            'rt_user_col': rt_user_col,
            'rt_item_fk_col': rt_item_fk_col,
            'rt_rating_col': rt_rating_col,
            'item_table': it,
            'it_item_id_col': it_item_id_col,
            'it_item_mean_col': it_item_mean_col,
            'user_table': ut,
            'ut_id_col': ut_id_col
        }
        self.normalize = normalize
    
    
    def get_conn(self):
        conn = sqlite3.connect(self.sql3_f)
        conn.row_factory = sqlite3.Row
        return conn
    
    @staticmethod
    def _sqlite_rows_2_dataframe(rows, columns):
        data = defaultdict(list)
        for row in rows:
            cols = rows[0].keys()
            for col,v in zip(cols, row):  data[col].append(v)
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
        ret = ExplicitDataFromSql3._sqlite_rows_2_dataframe(res, columns=[rt_ifk, rt_r])
        normalize = self.normalize
        if normalize is not None: ret[rt_r] = (ret[rt_r]-normalize['loc'])/normalize['scale']
        return ret

    def make_training_datasets(self, dtype:str='dense', ctype:np.dtype=np.int32, rtype:np.dtype=np.float64, users:Optional[Iterable[int]]=None):
        
        if users is None: users = np.arange(self.N)
        assert len(users) > 0

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
            cond_str = ['{rt_u}={sub}'.format(rt_u=rt_u, sub=self.sub)]*len(users)
            cond_str = "(" + " OR ".join(cond_str) + ")"
            cmd += ' WHERE ' + cond_str
            cur.execute(cmd, tuple(users))
            rows = cur.fetchall()
        else:
            cur.execute(cmd)
            rows = cur.fetchall()
        
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

        normalize = self.normalize
        if normalize is not None: D['rating'] = (D['rating']-normalize['loc'])/normalize['scale']

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
        cur.execute('SELECT COUNT(*) FROM {it}'.format(it=it))
        res = cur.fetchall()
        conn.commit()
        return ExplicitDataFromSql3._sqlite_rows_2_dataframe(res, ['COUNT(*)']).iloc[0]['COUNT(*)']

    @property
    def N(self):
        conn = self.get_conn()
        cur = conn.cursor()
        ut  = self.conf['user_table']  
        ut_id_col = self.conf['ut_id_col']
        cur.execute('SELECT MAX({ut_id_col}) FROM {ut}'.format(ut_id_col=ut_id_col, ut=ut))
        res = cur.fetchall()
        conn.commit()
        colname = 'MAX({ut_id_col})'.format(ut_id_col=ut_id_col)
        return ExplicitDataFromSql3._sqlite_rows_2_dataframe(res, [colname]).iloc[0][colname] + 1 + self.Noffset
    
    @property
    def sub(self):
        return '?'

    def import_items(self, **kwargs):
        """
        Utility function for importing stuff from ExplicitDataFromCSV (or others) into ExplicitDataFromSql3
        """
        assert not self.normalize, "cannot alter data in normalized mode. please set normalize to None"
        assert 'item_ids' in kwargs, "required: item_ids"
        assert 'names' in kwargs, "required: names"
        assert 'mean_rating' in kwargs, "required: mean_rating"
        conn = self.get_conn()
        cur = conn.cursor()

        it  = self.conf['item_table']
        it_i  = self.conf['it_item_id_col']


        insert_str = 'INSERT INTO {it} ({it_i}, created_at, name, \"desc\", poster, mean_rating) VALUES ({sub}, {sub}, {sub}, {sub}, {sub}, {sub}) '.format(it=it, it_i=it_i, sub=self.sub)

        cur.executemany(insert_str, [(idx, '2020-05-30 20:43:24.088110', name, desc, poster, mean_rating) for (idx, name, desc, poster, mean_rating) in zip(kwargs['item_ids'], kwargs['names'], kwargs['desc'], kwargs['poster_path'], kwargs['mean_rating'])])
        conn.commit()
    
        
    def get_item_mean_ratings(self, items=None):

        conn = self.get_conn()
        cur = conn.cursor()
        
        it  = self.conf['item_table']
        it_i  = self.conf['it_item_id_col']
        it_m = self.conf['it_item_mean_col']

        cmd = 'SELECT {it_i}, {it_m} FROM {it}'.format(it_m=it_m, it_i=it_i, it=it)
        cond_str = ""

        if items is not None:
            cond_str = ['{it_i}={sub}'.format(it_i=it_i, sub=self.sub)]*len(items)
            cond_str = "(" + " OR ".join(cond_str) + ")"
            cmd += ' WHERE ' + cond_str
            cur.execute(cmd, tuple(items))
            rows = cur.fetchall()
        else:
            cur.execute(cmd)
            rows = cur.fetchall()
        
        mean_df = ExplicitDataFromSql3._sqlite_rows_2_dataframe(rows, [it_i, it_m]).set_index([it_i])
        mean_df.rename(columns={it_m: 'rating_item_mean'}, inplace=True)
        
        normalize = self.normalize
        if normalize is not None: mean_df['rating_item_mean'] = (mean_df['rating_item_mean']-normalize['loc'])/normalize['scale']
        return mean_df


class ExplicitDataFromPostgres(ExplicitDataFromSql3):

    def __init__(self, postgres_config: str, **kwargs):
        with open(postgres_config, 'r') as fp: self.postgres_config = fp.read()
        super(ExplicitDataFromPostgres, self).__init__(None, **kwargs)
        self.conn = None

    def get_conn(self):
        if self.conn is None:
            self.conn = psycopg2.connect(self.postgres_config, cursor_factory=psycopg2.extras.DictCursor)
        return self.conn
    
    @property
    def M(self):
        conn = self.get_conn()
        cur = conn.cursor()
        it  = self.conf['item_table']        
        cur.execute('SELECT COUNT(*) FROM {it}'.format(it=it))
        res = cur.fetchall()
        conn.commit()
        return ExplicitDataFromSql3._sqlite_rows_2_dataframe(res, ['count']).iloc[0]['count']

    @property
    def N(self):
        conn = self.get_conn()
        cur = conn.cursor()
        ut  = self.conf['user_table']  
        ut_id_col = self.conf['ut_id_col']
        cur.execute('SELECT MAX({ut_id_col}) FROM {ut}'.format(ut_id_col=ut_id_col, ut=ut))
        res = cur.fetchall()
        conn.commit()
        colname = 'max'
        return ExplicitDataFromSql3._sqlite_rows_2_dataframe(res, [colname]).iloc[0][colname] + 1 + self.Noffset
    
    @property
    def sub(self):
        return '%s'

class ExplicitDataFromCSV(ExplicitData):

    def from_raw_csv(self, ratings_csv, r_item_col, r_user_col, r_rating_col, metadata_csv, m_item_col, train_test_split=0.8):
        self.ratings, self.user_map, self.item_map = \
            ExplicitDataFromCSV.sanitize_ratings(ratings_csv, r_item_col, r_user_col, r_rating_col)
        _, test_split = splitDf(ratings_csv, train_test_split)
        self.ratings['split'] = 0
        self.ratings.iloc[test_split]['split'] = 1
        self.md_df = ExplicitDataFromCSV.makeMetadataDf(self.item_map.copy(), metadata_csv, m_item_col)
        self.stats = ExplicitDataFromCSV.calc_rating_stats(self.ratings)

    def from_saved_csv(self, ratings_csv, split_csv, user_map_csv, item_map_csv, md_csv, stats_csv):
        if ratings_csv is not None:
            self.ratings = pd.read_csv(ratings_csv, index_col=None)[['rating', 'item', 'user']]
            self.ratings_split  = pd.read_csv(split_csv, index_col=None)
            assert len(self.ratings) == len(self.ratings_split)
            self.ratings['split'] = self.ratings_split
            self.ratings.set_index(['user', 'split'], drop=False, inplace=True)
        self.user_map = pd.read_csv(user_map_csv, index_col='user_cat')
        self.item_map = pd.read_csv(item_map_csv, index_col='item_cat')
        self.md_df = pd.read_csv(md_csv, index_col='item_cat')
        self.stats = pd.read_csv(stats_csv, index_col='item')

    def from_standard_dataset(self, data_folder, load_ratings=True):
        self.from_saved_csv(
            ratings_csv=os.path.join(data_folder, 'ratings_sanitized.csv') if load_ratings else None,
            split_csv=os.path.join(data_folder, 'ratings_split.csv') if load_ratings else None,
            user_map_csv = os.path.join(data_folder, 'user_map.csv'),
            item_map_csv = os.path.join(data_folder, 'item_map.csv'),
            md_csv = os.path.join(data_folder, 'metadata.csv'),
            stats_csv = os.path.join(data_folder, 'stats.csv'),
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

        self.normalize = None
        return

    def fetch_md(self, item_ids):
        return self.md_df.loc[item_ids]

    def get_ratings(self):
        return self.ratings

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
        
    def get_ratings_split(self, split):
        if split not in self.ratings.index: return pd.DataFrame({}, columns=self.ratings.columns)
        return self.ratings.loc[self.ratings.split==split]

    @staticmethod
    def sanitize_ratings(ratings_csv, item, user, rating):
        return normalizeDf(csv2df(ratings_csv, item, user, rating))

    @staticmethod
    def calc_rating_stats(df):
        rating_stats = df.groupby('item', as_index=False).count()[['item', 'user']]
        rating_stats.set_index('item', inplace=True)
        rating_stats.rename({'user': 'r_count'}, inplace=True)
        rating_item_mean = df.groupby('item', as_index=False).mean()[['rating']]
        rating_stats['rating_item_mean'] = rating_item_mean
        return rating_stats

    @staticmethod
    def makeMetadataDf(item_map, item_md_csv, item_id_col):
        item_md_df = pd.read_csv(item_md_csv, index_col=item_id_col)
        item_map = item_map.merge(item_md_df, left_index=True, right_index=True)
        item_map.set_index('item_cat', inplace=True)
        item_map.rename({'item_cat':'item'}, axis=1, inplace=True)
        return item_map

    def save(self, dir):
        assert not self.normalize, "cannot sasve in normalized mode. please set normalize to None"
        self.ratings[['rating', 'item', 'user']].to_csv(os.path.join(dir, 'ratings_sanitized.csv'), index=False)
        self.ratings[['split']].to_csv(os.path.join(dir, 'ratings_split.csv'), index=False)
        self.user_map.to_csv(os.path.join(dir, 'user_map.csv'))
        self.item_map.to_csv(os.path.join(dir, 'item_map.csv'))
        self.md_df.to_csv(os.path.join(dir, 'metadata.csv'))
        self.stats.to_csv(os.path.join(dir, 'stats.csv'))

    def make_training_datasets(self, dtype:str='dense', ctype:np.dtype=np.int32, rtype:np.dtype=np.float64, users:Optional[Iterable[int]]=None):
        if users is not None:
            D_train = self.get_user_ratings(users, train=True)
            D_test = self.get_user_ratings(users, train=False)
        else:
            D_train, D_test = self.df_train, self.df_test
        assert dtype in {'dense', 'sparse', 'df'}, "dtype not understood"
        if dtype == 'dense':
            return ExplicitDataFromCSV.make_positive_arrays(D_train, ctype, rtype), \
                   ExplicitDataFromCSV.make_positive_arrays(D_test, ctype, rtype)
        elif dtype=='df':
            return D_train, D_test
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
    
    def get_item_mean_ratings(self, items):
        if items is None: return self.stats.loc[:, 'rating_item_mean']
        return self.stats.loc[items, ['rating_item_mean']]

class ExplicitDataDummy(ExplicitData):

    def __init__(self, m=10, n=20, nblocks=10):

        super(ExplicitDataDummy, self).__init__()

        M = m*nblocks
        N = n*nblocks

        self._M = M
        self._N = N
        train_val_split = 0.8


        data = np.array([])
        cols = np.array([])
        rows = np.array([])

        for b in range(nblocks):
            bcols = np.arange(m*b, m*(b+1))
            brows = np.arange(n*b, n*(b+1))
            bcols = np.tile(bcols, (n,1)).transpose().flatten()
            brows = np.tile(brows, (m,1)).flatten()
            cols = np.concatenate([cols, bcols])
            rows = np.concatenate([rows, brows])
            data = np.concatenate([data, np.ones(shape=(m * n)) * (b%5+1)])

        perm = np.random.permutation(np.arange(len(data)))
        data = data[perm]
        rows = rows[perm].astype(int)
        cols = cols[perm].astype(int)

        data_train = data[:int(train_val_split * len(data))]
        data_test = data[int(train_val_split * len(data)):]
        rows_train = rows[:int(train_val_split * len(data))]
        rows_test = rows[int(train_val_split * len(data)):]
        cols_train = cols[:int(train_val_split * len(data))]
        cols_test = cols[int(train_val_split * len(data)):]

        self.df_train = pd.DataFrame({'rating': data_train, 'item': cols_train, 'user': rows_train}).set_index('user', drop=False)
        self.df_test = pd.DataFrame({'rating': data_test, 'item': cols_test, 'user': rows_test}).set_index('user', drop=False)

    def fetch_md(self, item_id):
        pass

    def get_ratings(self):
        pass

    def make_training_datasets(self, dtype:str='dense', ctype:np.dtype=np.int32, rtype:np.dtype=np.float64, users:Optional[Iterable[int]]=None):

        if users is not None:
            D_train = self.get_user_ratings(users, train=False)
            D_test = self.get_user_ratings(users, train=False)
        else:
            D_train, D_test = self.df_train, self.df_test

        assert dtype in {'dense', 'sparse', 'df'}, "dtype not understood"
        if dtype == 'dense':
            return ExplicitDataFromCSV.make_positive_arrays(D_train, ctype, rtype), \
                   ExplicitDataFromCSV.make_positive_arrays(D_test, ctype, rtype)
        elif dtype=='df':
            return D_train, D_test
        elif dtype == 'sparse':
            return ExplicitDataFromCSV.make_sparse_matrices(D_train, self.N, self.M, ctype, rtype), \
                   ExplicitDataFromCSV.make_sparse_matrices(D_test, self.N, self.M, ctype, rtype)

    def get_user_ratings(self, users, train=False):
        if train: ratings = self.df_train
        else: ratings = self.df_test
        users = ratings.index.intersection(users)
        ratings = ratings.loc[users]

        return ratings

    @property
    def N(self):
        return self._N

    @property
    def M(self):
        return self._M

    @property
    def Nranked(self):
        return len(self.df_train) + len(self.df_test)

    def get_item_mean_ratings(self, items):
        # TODO
        raise NotImplementedError


if __name__=="__main__":

    df_train, df_test = ExplicitDataDummy().make_training_datasets(dtype='sparse')
    np.set_printoptions(threshold=np.inf)
    print(df_train.todense())

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
    ratings_all.reset_index(drop=True).to_csv('/home/ong/personal/recommender/data/ml-20m/ratings_sanitized.csv', index=False)
    df_split.to_csv('/home/ong/personal/recommender/data/ml-20m/ratings_split.csv', index=False)
    """

    # data_dir = "/home/ong/personal/recommender/data/ml-latest-small"
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