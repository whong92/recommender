import tensorflow as tf

# tensorflow setup shenanigans
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

from recommender.utils.ItemMetadata import ExplicitDataFromCSV
from recommender.recommender.recommenderALS import RecommenderALS

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

if __name__=="__main__":

    data_folder = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small'
    model_folder = 'D:\\PycharmProjects\\recommender\\models'

    d = ExplicitDataFromCSV(True, data_folder=data_folder)

    # remove training data for 10 users to test
    users_remove = np.arange(0, d.N, d.N//10)

    train_ratings_holdout = d.pop_user_ratings(users_remove)

    # train with rating removed
    save_path = os.path.join(model_folder, "ALS_{:s}".format(datetime.now().strftime("%Y-%m-%d.%H-%M-%S")))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    rals = RecommenderALS(mode='train', n_users=d.N, n_items=d.M,
                        als_kwargs={'K':20, 'lamb':1e-06, 'alpha':40., 'steps': 5},
                        model_path=save_path
                        )

    # array input format - onnly for smaller datasets
    rals.input_data(d)
    trace = rals.train()
    # plt.plot(trace)
    # plt.show()
    rals.save(save_path)

    # reload model
    rals = RecommenderALS(mode='predict', model_path=save_path)
    rals.input_data(d)
    # update model with removed ratings, something like this
    d.add_user_ratings(train_ratings_holdout['user'], train_ratings_holdout['item'], train_ratings_holdout['rating'])
    trace, auc = rals.train_update(users=users_remove)
    print(trace, auc)

    # add a user, add phony ratings
    d.add_user()
    new_ratings_train = train_ratings_holdout.loc[0].copy()
    new_ratings_train['user'] = d.N - 1
    new_ratings_test = d.get_user_ratings(0, train=False).copy()
    new_ratings_test['user'] = d.N - 1
    d.add_user_ratings(new_ratings_train['user'], new_ratings_train['item'], new_ratings_train['rating'], train=True)
    d.add_user_ratings(new_ratings_test['user'], new_ratings_test['item'], new_ratings_test['rating'], train=False)

    rals.add_users(1)
    trace, auc = rals.train_update(users=np.array([d.N - 1]))