from ..utils.utils import csv2df, splitDf
from ..utils.ItemMetadata import ExplicitDataFromCSV
from .recommenderALS import RecommenderALS
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import os

if __name__=="__main__":

    data_folder = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small'
    model_folder = 'D:\\PycharmProjects\\recommender\\recommender\\models'

    # TODO: separate the making of the dataset and using it in a separate script
    """
    ratings_csv = os.path.join(data_folder, 'ratings.csv')
    movies_csv = os.path.join(data_folder, 'movies.csv')
    d = ExplicitDataFromCSV(False,
                            ratings_csv=ratings_csv, r_item_col='movieId', r_user_col='userId',
                            r_rating_col='rating',
                            metadata_csv=movies_csv, m_item_col='movieId')
    """

    ratings_csv = os.path.join(data_folder, 'ratings_sanitized.csv')
    user_map_csv = os.path.join(data_folder, 'user_map.csv')
    item_map_csv = os.path.join(data_folder, 'item_map.csv')
    md_csv = os.path.join(data_folder, 'metadata.csv')
    stats_csv = os.path.join(data_folder, 'stats.csv')
    df_train = os.path.join(data_folder, 'ratings_train.csv')
    df_test = os.path.join(data_folder, 'ratings_test.csv')
    d = ExplicitDataFromCSV(True,
                            ratings_csv=ratings_csv,
                            user_map_csv=user_map_csv,
                            item_map_csv=item_map_csv,
                            md_csv=md_csv,
                            stats_csv=stats_csv,
                            ratings_train_csv = df_train,
                            ratings_test_csv = df_test
                            )
    d.save(data_folder)

    # training
    train_test_split = 0.8
    ds_train, ds_test, D_train, D_test = d.make_training_datasets(train_test_split)
    # D_train.to_csv(os.path.join(data_folder, 'ratings_train.csv'))
    # D_test.to_csv(os.path.join(data_folder, 'ratings_test.csv'))

    save_path = os.path.join(model_folder, "ALS_{:s}".format(datetime.now().strftime("%m%Y%d%H%M%S")))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # TODO: test with RecommenderMFBias
    rals = RecommenderALS(mode='train', n_users=d.N, n_items=d.M,
                        als_kwargs={'K':10, 'lamb':1e-06, 'alpha':40.},
                        model_path=save_path
                        )
    tf.logging.set_verbosity(tf.logging.INFO)

    # array input format - onnly for smaller datasets
    rals.input_array_data(
        np.array(D_train['user'], dtype=np.int32),
        np.array(D_train['item'], dtype=np.int32),
        np.array(D_train['rating'], dtype=np.float64),
        np.array(D_test['user'], dtype=np.int32),
        np.array(D_test['item'], dtype=np.int32),
        np.array(D_test['rating'], dtype=np.float64),
    )
    rals.train()

    rals.save(save_path)
