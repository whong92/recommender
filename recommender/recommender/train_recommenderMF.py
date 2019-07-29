from ..utils.utils import csv2df, splitDf
from ..utils.ItemMetadata import ExplicitDataFromCSV
from .recommenderMF import RecommenderMF, RecommenderMFBias
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import os

if __name__=="__main__":

    data_folder = 'D:\\PycharmProjects\\recommender\\data\\ml-20m'
    model_folder = 'D:\\PycharmProjects\\recommender\\recommender\\models'

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
    d = ExplicitDataFromCSV(True,
                            ratings_csv=ratings_csv,
                            user_map_csv=user_map_csv,
                            item_map_csv=item_map_csv,
                            md_csv=md_csv,
                            stats_csv=stats_csv)
    d.save(data_folder)

    # training
    train_test_split = 0.8
    ds_train, ds_test, D_train, D_test = d.make_training_datasets(train_test_split)
    D_train.to_csv(os.path.join(data_folder, 'ratings_train.csv'))
    D_test.to_csv(os.path.join(data_folder, 'ratings_test.csv'))

    save_path = os.path.join(model_folder, datetime.now().strftime("%m%Y%d%H%M%S"))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    rmf = RecommenderMF(mode='train', n_users=d.N, n_items=d.M,
                        mf_kwargs={'f':20,'lamb':1e-06},
                        model_path=save_path
                        )
    tf.logging.set_verbosity(tf.logging.INFO)

    # array input format - onnly for smaller datasets
    rmf.input_data(ds_train, ds_test)
    rmf.train(numepochs=10, batchsize=5000)

    rmf.save(save_path)
