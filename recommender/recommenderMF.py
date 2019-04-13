from ..recommenderInterface import Recommender
from .MatrixFactor import MatrixFactorizer
from ..hash.LSH import LSH
from ..hash.Signature import MakeSignature
from ..utils.utils import csv2df, splitDf
from tensorflow.contrib import predictor
import numpy as np
import scipy.sparse as sps
import pandas as pd
import tensorflow as tf
import os


class recommenderMF(Recommender):

    def __init__(self, n_users, n_items, mode='train', model_path='.', lsh_path='.'):

        self.mode = mode
        self.estimator = None
        self.predictor = None
        self.n_users = n_users
        self.n_items = n_items
        self.lsh = None

        if mode is 'train':
            f = 20  # latent factor dimensionality
            lamb = 0.001
            lr = 0.01
            decay = 0.0
            self.estimator = MatrixFactorizer(n_users, n_items, f, lr, lamb, decay)
        elif mode is 'predict':
            sig_path = os.path.join(lsh_path, 'signature.json')
            hash_path = os.path.join(lsh_path, 'hash.json')
            self.lsh = LSH(MakeSignature('CosineHash', path=sig_path), path=hash_path)
            self.predictor = predictor.from_saved_model(model_path)

    def train(self, u_in, i_in, r_in, u_in_test, i_in_test, r_in_test, model_path, lsh_path):

        assert self.mode is 'train', "must be in train mode!"

        self.estimator.fit(
            {'u_in': u_in, 'i_in': i_in},
            r_in,
            {'u_in': u_in_test, 'i_in': i_in_test},
            r_in_test,
            numepochs=30,
        )

        # generate embeddings for all items, and insert into LSH
        pred = predictor.from_estimator(self.estimator.model, MatrixFactorizer._predict_input_fn)
        p = pred(
            {'u_in': np.zeros(shape=(self.n_items,), dtype=np.int32),
             'i_in': np.arange(1, self.n_items+1, dtype=np.int32)
             })

        sig = MakeSignature('CosineHash', num_row=20, num_hpp=100)
        sig.save(os.path.join(lsh_path, 'signature.json'))
        self.lsh = LSH(sig, num_bands=5)
        self.lsh.insert(sps.csc_matrix(p['q'].transpose()))
        self.lsh.save(os.path.join(lsh_path, 'hash.json'))
        self.estimator.save(model_path)

    def make_and_update_hash(self, lsh_path=None):
        if self.mode is'train':
            pred = predictor.from_estimator(self.estimator.model, MatrixFactorizer._predict_input_fn)
        else:
            pred = self.predictor
        p = pred(
            {'u_in': np.zeros(shape=(self.n_items,), dtype=np.int32),
             'i_in': np.arange(1, self.n_items + 1, dtype=np.int32)
             })
        sig = MakeSignature('CosineHash', num_row=20, num_hpp=400)
        self.lsh = LSH(sig, num_bands=10)
        self.lsh.insert(sps.csc_matrix(p['q'].transpose()))
        if lsh_path is not None:
            sig.save(os.path.join(lsh_path, 'signature.json'))
            self.lsh.save(os.path.join(lsh_path, 'hash.json'))


    def predict(self, u_in, i_in):

        assert self.mode is 'predict'
        return self.predictor({'u_in': u_in, 'i_in': i_in})

    def recommend_lsh(self, u_in):

        assert type(u_in) is int

        p = self.predictor({'u_in': np.array([u_in], dtype=np.int32),
                        'i_in': np.array([0], dtype=np.int32)})


        bla = self.lsh.find_similar(
            sps.csc_matrix(np.expand_dims(p['p'][0,:], axis=1))
        )
        idx = np.array(list(bla), dtype=np.int32)
        p = self.predictor({'u_in': np.tile(np.array([u_in], dtype=np.int32), len(idx)),
                            'i_in': idx})
        return idx[np.argsort(p['rhat'])] + 1, np.sort(p['rhat'])

    def recommend(self, u_in):
        # manual prediction by enumerating all stuff
        p = self.predictor({'u_in': np.tile(np.array([u_in], dtype=np.int32), self.n_items),
                            'i_in': np.arange(1, self.n_items + 1, dtype=np.int32)})

        return np.argsort(p['rhat']) + 1, np.sort(p['rhat'])



if __name__=="__main__":

    df, user_map, item_map, N, M = csv2df('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv',
                      'movieId', 'userId', 'rating', return_cat_mapping=True)
    """
    # training
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
        model_path = './bla',
        lsh_path = './bla'
    )

    """

    # prediction
    df_mov = pd.read_csv('D:/PycharmProjects/recommender/data/ml-latest-small/movies.csv')
    rmf = recommenderMF(N, M, mode='predict', model_path='./bla/1555182808', lsh_path='./bla')
    print(np.max(df.item))
    rmf.make_and_update_hash(lsh_path='./bla')
    user = 99
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

    import matplotlib.pyplot as plt
    rating_counts = np.flip(np.sort(np.array(df.groupby('item').count()['user'], dtype=np.int32)))
    plt.subplot('211')
    plt.plot(np.linspace(0, len(rating_counts), len(rating_counts), dtype=np.int32), rating_counts, 'o')
    plt.subplot('212')
    plt.plot(np.linspace(0, len(rating_counts), len(rating_counts), dtype=np.int32), np.cumsum(rating_counts), 'o')
    #plt.hist(user_df.rhat)
    plt.show()

    print(sorted_items.merge(df_mov, left_on='item', right_on='movieId')[:10])
    print(user_df.sort_values('rating').merge(df_mov, left_on='item', right_on='movieId')[:10])

    print(sorted_items.merge(df_mov, left_on='item', right_on='movieId')[-10:])
    print(user_df.sort_values('rating').merge(df_mov, left_on='item', right_on='movieId')[-10:])








