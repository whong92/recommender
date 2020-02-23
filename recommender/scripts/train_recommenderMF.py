from recommender.utils.utils import csv2df, splitDf
from recommender.utils.ItemMetadata import ExplicitDataFromCSV
from recommender.recommender.recommenderMF import RecommenderMF
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import os

if __name__=="__main__":

    data_folder = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small'
    model_folder = 'D:\\PycharmProjects\\recommender\\models'

    # TODO: separate the making of the dataset and using it in a separate script
    """
    ratings_csv = os.path.join(data_folder, 'ratings.csv')
    movies_csv = os.path.join(data_folder, 'movies.csv')
    d = ExplicitDataFromCSV(False,
                            ratings_csv=ratings_csv, r_item_col='movieId', r_user_col='userId',
                            r_rating_col='rating',
                            metadata_csv=movies_csv, m_item_col='movieId')
    ds_train, ds_test, D_train, D_test = d.make_training_datasets(train_test_split)
    d.save(data_folder)
    """

    d = ExplicitDataFromCSV(True, data_folder=data_folder)
    d.save(data_folder)

    # training
    train_test_split = 0.8
    save_path = os.path.join(model_folder, "MF_{:s}".format(datetime.now().strftime("%Y-%m-%d.%H-%M-%S")))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # TODO: test with RecommenderMFBias
    rmf = RecommenderMF(mode='train', n_users=d.N, n_items=d.M,
                        mf_kwargs={
                            'f':20,'lamb':1e-07, 'decay':0.9, 'lr':0.005,
                            'epochs': 50, 'batchsize': 5000,
                        },
                        model_path=save_path
                        )
    # tf.logging.set_verbosity(tf.logging.INFO)

    # array input format - onnly for smaller datasets
    rmf.input_data(d)
    rmf.train()

    rmf.save(save_path)
