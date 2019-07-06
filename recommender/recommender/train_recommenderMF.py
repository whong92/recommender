from ..utils.utils import csv2df, splitDf
from .recommenderMF import RecommenderMF
import tensorflow as tf
from ..utils.mongodbutils import DataService
from ..hash.LSH import MakeLSH
from ..hash.Signature import MakeSignature
import numpy as np
import pandas as pd
from datetime import datetime
import os

if __name__=="__main__":

    df, user_map, item_map, N, M = csv2df('D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\ratings.csv',
                                          'movieId', 'userId', 'rating', return_cat_mapping=True)

    # training
    train_test_split = 0.8
    D_train, D_test = splitDf(df, train_test_split)
    D_train.to_csv('D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\ratings_train.csv')
    D_test.to_csv('D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\ratings_test.csv')


    Users_train = D_train['user']
    Items_train = D_train['item']
    Ratings_train = D_train['rating']

    print(len(D_train['rating']))

    Users_test = D_test['user']
    Items_test = D_test['item']
    Ratings_test = D_test['rating']
    """
    user_map = pd.DataFrame.from_csv("D:\\PycharmProjects\\recommender\\data\\tmp\\user_map_df.csv")
    item_map = pd.DataFrame.from_csv("D:\\PycharmProjects\\recommender\\data\\tmp\\item_map_df.csv")
    """
    N = len(user_map)
    M = len(item_map)

    #dbconn = ("localhost", 27017)
    #db_name = "test"
    #url = "mongodb://{:s}:{:d}/".format(*dbconn)
    #ds = DataService(url, db_name)
    #sig = MakeSignature("CosineHash", num_row=10)
    #lsh = MakeLSH("LSHDB", sig=sig, data_service=ds, pref='hullabalooza', num_bands=5)
    lsh = None
    rmf = RecommenderMF(mode='train', n_users=N, n_items=M,
                        mf_kwargs={'f':10,'lamb':.005},
                        lsh=lsh
                        )
    tf.logging.set_verbosity(tf.logging.INFO)

    # array input format - onnly for smaller datasets
    rmf.input_array_data(
        np.array(Users_train, dtype=np.int32),
        np.array(Items_train, dtype=np.int32),
        np.array(Ratings_train, dtype=np.float64),
        np.array(Users_test, dtype=np.int32),
        np.array(Items_test, dtype=np.int32),
        np.array(Ratings_test, dtype=np.float64),
    )

    # save train-test csvs
    D_train.to_csv('D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\ratings_train.csv')
    D_test.to_csv('D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\ratings_train.csv')

    """
    rmf.input_tfr_paths(
        ['D:/PycharmProjects/recommender/data/tmp-20m/bla_train{:03d}.tfrecord'.format(i) for i in range(1)],
        ['D:/PycharmProjects/recommender/data/tmp-20m/bla_test{:03d}.tfrecord'.format(i) for i in range(1)]
    )
    """

    rmf.train()
    save_path = os.path.join('D:/PycharmProjects/recommender/recommender/models/model/', datetime.now().strftime("%m%Y%d%H%M%S"))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    rmf.save(save_path)

    df_mov = pd.read_csv('D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\movies.csv')
    df_mov = df_mov.merge(item_map, left_on='movieId', right_on='item')
    df_mov = df_mov.merge(pd.DataFrame({'r_count': df.groupby('item').count()['user']})
                          , left_on='item_cat', right_on='item')
    #ds.insert_movie_mds('hullabalooza', df_mov.title, df_mov.movieId, df_mov.item_cat, df_mov.r_count)