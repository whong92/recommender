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
    save_path = os.path.join(model_folder, "MF_2020-05-30.23-40-30")
    save_path_asvdc = os.path.join(save_path, "ASVDC")
    if not os.path.exists(save_path_asvdc): os.mkdir(save_path_asvdc)
    aucc = AUCCallback(
        data=d, outfile=os.path.join(save_path, 'AUC.csv'), 
        M=np.arange(0, d.N, d.N//300,dtype=int),
        save_fn=None, batchsize=10
    )

    models = list(filter(lambda x: x.endswith(".h5"), os.listdir(save_path)))

    for i, m in enumerate(models):
        f = parse.parse(MatrixFactorizer.MODEL_FORMAT, m)
        if f is None: continue
        epoch = f['epoch']

        rmf = RecommenderMFAsym(
            mode='predict', n_users=d.N, n_items=d.M, n_ranked=d.Nranked,
            mf_kwargs={
                'config_path': 'SVD_asym.json.template'
            },
            model_path=save_path, saved_model=os.path.join(save_path, m)
        )
        rmf.input_data(d)

        rmfa = RecommenderMFAsymCached(
            mode='train', n_users=d.N, n_items=d.M, n_ranked=d.Nranked,
            mf_kwargs={
                'config_path': 'SVD_asym_cached.json.template'
            },
            model_path=save_path_asvdc, saved_model=None
        )

        rmfa.input_data(d)
        rmfa.import_asym(rmf)
        rmfa.save(model_path=m)

        print('auc rmfa')
        aucc.set_model(rmfa)
        aucc.on_epoch_end(epoch)

        print('eval on rmf')
        rmf.evaluate()

        print('eval on rmfa')
        rmfa.evaluate()