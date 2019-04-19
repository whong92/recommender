from ..utils.utils import csv2df, splitDf
from .recommenderMF import recommenderMF
import tensorflow as tf
import numpy as np

if __name__=="__main__":
    df, user_map, item_map, N, M = csv2df('/home/ong/Downloads/ml-latest-small/ratings.csv',
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

    rmf = recommenderMF(mode='train', n_users=N, n_items=M)
    tf.logging.set_verbosity(tf.logging.INFO)
    rmf.train(
        np.array(Users_train, dtype=np.int32),
        np.array(Items_train, dtype=np.int32),
        np.array(Ratings_train, dtype=np.float64),
        np.array(Users_test, dtype=np.int32),
        np.array(Items_test, dtype=np.int32),
        np.array(Ratings_test, dtype=np.float64),
        model_path='./bla',
        lsh_path='./bla'
    )