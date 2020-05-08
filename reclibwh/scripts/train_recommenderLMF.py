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

from reclibwh.utils.ItemMetadata import ExplicitDataFromCSV
from reclibwh.recommender.recommenderLMF import RecommenderLMF

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

if __name__=="__main__":

    # data_folder = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small'
    # model_folder = 'D:\\PycharmProjects\\recommender\\models'
    data_folder = '/home/ong/personal/recommender/data/ml-latest-small'
    model_folder = '/home/ong/personal/recommender/models'

    d = ExplicitDataFromCSV(True, data_folder=data_folder)
    # d.save(data_folder)

    save_path = os.path.join(model_folder, "LMF_2020-04-13.17-41-24")#, "LMF_{:s}".format(datetime.now().strftime("%Y-%m-%d.%H-%M-%S")))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    rlmf = RecommenderLMF(
        mode='train', n_users=d.N, n_items=d.M,
        lmf_kwargs={'f':10, 'lamb':1e-06, 'alpha':5., 'lr': 0.1, 'bias': False, 'epochs':5},
        model_path=save_path
    )
    # tf.logging.set_verbosity(tf.logging.INFO)

    # array input format - only for smaller datasets
    rlmf.input_data(d)
    trace = rlmf.train()
    plt.plot(trace)
    plt.show()

    rlmf.save(save_path)
