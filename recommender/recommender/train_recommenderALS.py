import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from ..utils.ItemMetadata import ExplicitDataFromCSV
from .recommenderALS import RecommenderALS

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

if __name__=="__main__":

    data_folder = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small'
    model_folder = 'D:\\PycharmProjects\\recommender\\models'

    # TODO: separate the making of the dataset and using it in a separate script
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
    _, _, D_train, D_test = d.make_training_datasets(train_test_split)

    save_path = os.path.join(model_folder, "ALS_{:s}".format(datetime.now().strftime("%m%Y%d%H%M%S")))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    rals = RecommenderALS(mode='train', n_users=d.N, n_items=d.M,
                        als_kwargs={'K':10, 'lamb':1e-06, 'alpha':40.},
                        model_path=save_path
                        )
    # tf.logging.set_verbosity(tf.logging.INFO)

    # array input format - onnly for smaller datasets
    rals.input_array_data(
        np.array(D_train['user'], dtype=np.int32),
        np.array(D_train['item'], dtype=np.int32),
        np.array(D_train['rating'], dtype=np.float64),
        np.array(D_test['user'], dtype=np.int32),
        np.array(D_test['item'], dtype=np.int32),
        np.array(D_test['rating'], dtype=np.float64),
    )

    trace = rals.train()
    plt.plot(trace)
    plt.show()

    rals.save(save_path)
