import tensorflow as tf

# tf.config.experimental.set_visible_devices([], 'GPU')
# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

from reclibwh.utils.ItemMetadata import ExplicitDataFromCSV
from reclibwh.recommender.recommenderMF import RecommenderMF, RecommenderMFAsym, RecommenderMFAsymCached
from reclibwh.core.MatrixFactor import MatrixFactorizer
from reclibwh.core.AsymSVD import AsymSVD, AsymSVDCached
from reclibwh.utils.eval_utils import AUCCallback
from datetime import datetime
import numpy as np
import os
import parse
from ..utils.utils import get_pos_ratings_padded

if __name__=="__main__":

    data_folder = '/home/ong/personal/recommender/data/ml-latest-small-2'
    model_folder = '/home/ong/personal/recommender/models'

    d = ExplicitDataFromCSV(True, data_folder=data_folder, normalize={'loc': 0.0, 'scale': 5.0})

    # training
    save_path_asvdc = os.path.join(model_folder, "MF_2020-05-29.22-49-23", "ASVDC")
    if not os.path.exists(save_path_asvdc): os.mkdir(save_path_asvdc)

    saved_model = os.path.join(save_path_asvdc, 'model_best.h5')
    rmfa = RecommenderMFAsymCached(
        mode='predict', n_users=d.N, n_items=d.M, n_ranked=d.Nranked,
        mf_kwargs={
            'config_path_X': '/home/ong/personal/recommender/reclibwh/core/model_templates/SVD_asym_X.json.template',
            'config_path_main': '/home/ong/personal/recommender/reclibwh/core/model_templates/SVD_asym_cached.json.template'
        },
        model_path=save_path_asvdc, saved_model=saved_model
    )
    rmfa.input_data(d)

    d.set_normalize(None)
    d.add_user()
    (user_train, items_train, ratings_train), (user_test, items_test, ratings_test) = d.make_training_datasets(users=[0])
    user_train[:] = d.N-1
    user_test[:] = d.N-1
    d.add_user_ratings(user_train, items_train, ratings_train, train=True)
    d.add_user_ratings(user_test, items_test, ratings_test, train=False)
    d.set_normalize({'loc': 0.0, 'scale': 5.0})

    rmfa.add_users(1)
    rmfa.train_update(users=np.array([d.N-1]))