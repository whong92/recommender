from ..recommenderInterface import Recommender
from .MatrixFactor import MatrixFactorizer
from ..hash.LSH import LSH
from ..hash.Signature import MakeSignature
from ..utils.utils import csv2df, splitDf
from tensorflow.contrib import predictor
import numpy as np
import pandas as pd
import scipy.sparse as sps
import tensorflow as tf


class recommenderMF(Recommender):

    def __init__(self, n_users, n_items, mode='train', save_path='.'):

        self.mode = mode
        self.estimator = None
        self.predictor = None
        self.n_users = n_users
        self.n_items = n_items

        if mode is 'train':
            f = 20  # latent factor dimensionality
            lamb = 0.001
            lr = 0.01
            decay = 0.0
            self.estimator = MatrixFactorizer(n_users, n_items, f, lr, lamb, decay)
        elif mode is 'predict':
            self.predictor = predictor.from_saved_model(save_path)

    def train(self, u_in, i_in, r_in, u_in_test, i_in_test, r_in_test, save_path):

        assert self.mode is 'train', "must be in train mode!"

        self.estimator.fit(
            {'u_in': u_in, 'i_in': i_in},
            r_in,
            {'u_in': u_in_test, 'i_in': i_in_test},
            r_in_test,
            numepochs=20,
        )
        self.estimator.save(save_path)

    def predict(self, u_in, i_in):

        assert self.mode is 'predict'

        return self.predictor({'u_in': u_in, 'i_in': i_in})

    def recommend(self, u_in):

        assert type(u_in) is int

        p = self.predictor({'u_in': np.tile(np.array([u_in], dtype=np.int32), self.n_items),
                        'i_in': np.arange(1, self.n_items+1, dtype=np.int32)})
        lsh = LSH(MakeSignature('CosineHash', num_row=20, num_hpp=300))
        lsh.insert(sps.csc_matrix(p['q'].transpose()))
        bla = lsh.find_similar(
            sps.csc_matrix(np.expand_dims(p['p'][0,:], axis=1))
        )
        idx = np.array(list(bla), dtype=np.int32)
        p = self.predictor({'u_in': np.tile(np.array([u_in], dtype=np.int32), len(idx)),
                            'i_in': idx})
        return idx[np.argsort(p['rhat'])] + 1, np.sort(p['rhat'])


if __name__=="__main__":

    df, user_map, item_map, N, M = csv2df('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv',
                      'movieId', 'userId', 'rating', return_cat_mapping=True)

    """
    train_test_split = 0.8
    D_train, D_test = splitDf(df, train_test_split)

    Users_train = D_train['user']
    Items_train = D_train['item']
    Ratings_train = D_train['rating']

    print(len(D_train['rating']))

    Users_test = D_test['user']
    Items_test = D_test['item']
    Ratings_test = D_test['rating']

    rmf = recommenderMF(N, M, mode='train')
    tf.logging.set_verbosity(tf.logging.INFO)
    rmf.train(
        np.array(Users_train, dtype=np.int32),
        np.array(Items_train, dtype=np.int32),
        np.array(Ratings_train, dtype=np.float64),
        np.array(Users_test, dtype=np.int32),
        np.array(Items_test, dtype=np.int32),
        np.array(Ratings_test, dtype=np.float64),
        save_path = '.'
    )

    """
    df_mov = pd.read_csv('D:/PycharmProjects/recommender/data/ml-latest-small/movies.csv')
    rmf = recommenderMF(N, M, mode='predict', save_path='./1554996920')

    user = 567
    user_df = df.loc[df.user_cat==user]

    pred = rmf.predict(np.array(user_df['user']), np.array(user_df['item']))
    user_df['rhat'] = pred['rhat']
    user_df.rename(columns={'item':'item_code'}, inplace=True)
    user_df = user_df.merge(item_map, left_on='item_code', right_on='item_cat')

    sorted_items, sorted_score = rmf.recommend(user)
    sorted_items = pd.DataFrame(
        {
            'item_code': sorted_items,
            'rhat': sorted_score,
        }
    )
    sorted_items = sorted_items.merge(item_map, left_on='item_code', right_on='item_cat')

    print(sorted_items.merge(df_mov, left_on='item', right_on='movieId')[:10])
    print(user_df.sort_values('rating').merge(df_mov, left_on='item', right_on='movieId')[:10])


    print(sorted_items.merge(df_mov, left_on='item', right_on='movieId')[-10:])
    print(user_df.sort_values('rating').merge(df_mov, left_on='item', right_on='movieId')[-10:])



