from ..utils.utils import csv2df, splitDf
from .recommenderMF import RecommenderMF
import tensorflow as tf
import numpy as np

if __name__=="__main__":
    """
    df, user_map, item_map, N, M = csv2df('/home/ong/Downloads/ml-latest-small/ratings.csv',
                                          'movieId', 'userId', 'rating', return_cat_mapping=True)
    """
    df, user_map, item_map, N, M = csv2df('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv',
                                          'movieId', 'userId', 'rating', return_cat_mapping=True)

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

    rmf = RecommenderMF(mode='train', n_users=N, n_items=M)
    tf.logging.set_verbosity(tf.logging.INFO)
    """
    rmf.input_array_data(
        np.array(Users_train, dtype=np.int32),
        np.array(Items_train, dtype=np.int32),
        np.array(Ratings_train, dtype=np.float64),
        np.array(Users_test, dtype=np.int32),
        np.array(Items_test, dtype=np.int32),
        np.array(Ratings_test, dtype=np.float64),
    )
    """
    rmf.input_tfr_paths(
        ['D:/PycharmProjects/recommender/data/tmp/bla_train000.tfrecord'],
        ['D:/PycharmProjects/recommender/data/tmp/bla_test000.tfrecord']
    )
    rmf.train(
        model_path='./bla',
        lsh_path='./bla'
    )