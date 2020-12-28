import unittest
from reclibwh.utils.ItemMetadata import ExplicitDataDummy
from reclibwh.core.AsymSVD import AsymSVDEnv, AsymSVDCachedEnv
from reclibwh.data.PresetIterators import MFAsym_data_iter_preset, AUC_data_iter_preset
from reclibwh.core.Models import STANDARD_KERAS_SAVE_FMT, initialize_from_json
import os
import numpy as np
import pandas as pd
from reclibwh.utils.utils import mean_nnz
import scipy.sparse as sps
from reclibwh.utils.test_utils import refresh_dir

class TestAsymSVD(unittest.TestCase):

    def setUp(self) -> None:
        self.data = ExplicitDataDummy()
        self.save_path = "tests/models"

    def _test_asym_svd_train(self):

        d = self.data
        df_train, df_test = d.make_training_datasets(dtype='df')
        Utrain, Utest = d.make_training_datasets(dtype='sparse')
        M = d.M
        N = d.N
        Nranked = d.Nranked

        data_conf = {"M": M, "N": N + 1, "Nranked": Nranked}

        rnorm = {'loc': 0.0, 'scale': 5.0}
        data_train = MFAsym_data_iter_preset(df_train, Utrain, rnorm=rnorm)
        auc_test_data = AUC_data_iter_preset(Utest)
        auc_train_data = AUC_data_iter_preset(Utrain)

        data = {
            "train_data": data_train,
            "valid_data": data_train,
            "auc_data": {'test': auc_test_data, 'train': auc_train_data},
            "mf_asym_rec_data": {'U': Utrain, 'norm': rnorm},
        }
        env_vars = {"save_fmt": STANDARD_KERAS_SAVE_FMT, "data_conf": data_conf}
        m = initialize_from_json(data_conf=data_conf, config_path="SVD_asym.json.template", config_override={"SVD_asym": {"lr": 0.01}})[0]

        save_path = os.path.join(self.save_path, 'test_asym_svd_train')
        refresh_dir(save_path)
        env = AsymSVDEnv(save_path, m, data, env_vars, epochs=5, med_score=3.0)
        eval_res = env.fit()
        rmse = eval_res[2]
        self.assertLess(rmse, 0.01)

    def test_asym_svd_cached(self):

        # setup the reference AsymSVD model
        d = self.data
        rnorm = {'loc': 0.0, 'scale': 5.0}
        # df_train, _ = d.make_training_datasets(dtype='df')
        Utrain, Utest = d.make_training_datasets(dtype='sparse')

        data_conf = {"M": d.M, "N": d.N + 1, "Nranked": d.Nranked}
        m = initialize_from_json(data_conf=data_conf, config_path="SVD_asym.json.template")[0]
        save_path = os.path.join(self.save_path, 'test_asym_svd_cached_refmodel')
        env_vars = {"save_fmt": STANDARD_KERAS_SAVE_FMT, "data_conf": data_conf}
        env = AsymSVDEnv(save_path, m, {'mf_asym_rec_data': {'U': Utrain, 'norm': rnorm},}, env_vars, epochs=20, med_score=3.0)
        m = env['model']

        # initialize the cached model training
        save_path = os.path.join(self.save_path, 'test_asym_svd_cached_train')
        refresh_dir(save_path)
        mc = initialize_from_json(data_conf=data_conf, config_path="SVD_asym_cached.json.template")

        asvdc_data = MFAsym_data_iter_preset(
            pd.DataFrame({'user': np.arange(d.N), 'item': np.zeros(shape=(d.N,)), 'rating': np.zeros(shape=(d.N,))}),
            Utrain, rnorm=rnorm, remove_rated_items=False
        )
        env_varsc = {'data': {'asvd': m, 'train_data': asvdc_data}, 'data_conf': data_conf}
        envc = AsymSVDCachedEnv(save_path, mc, None, env_varsc)
        envc.fit()

        model_X, model_main = envc['model']

        asvd_X_weights = model_X.get_layer('X').get_weights()[0]
        asvdc_Q_weights = m.get_layer('Q').get_weights()[0]
        asvd_rec_0 = env.recommend([0])
        asvdc_rec_0 = envc.recommend([0])

        # check outputs match
        self.assertTrue(np.allclose(asvd_X_weights, asvdc_Q_weights))
        self.assertTrue(np.equal(asvd_rec_0[0], asvdc_rec_0[0]).all()) # item recs
        self.assertTrue(np.allclose(asvd_rec_0[1], asvdc_rec_0[1]))  # item scores

        # test update
        # use a cached version of the bias vector, removes the need to recompute from entire matrix
        Bi = np.reshape(np.array(mean_nnz(sps.csc_matrix(Utrain), axis=0, mu=0)), newshape=(-1,))
        df_train, df_test = d.make_training_datasets(dtype='df')
        # artificially add a user
        # add new user stuff
        df_update = df_train.loc[0].copy()
        df_update.loc[:, 'user'] = d.N
        df_update = df_update.reset_index(drop=True)
        vals = np.array(df_update['rating'])
        rows = np.array(df_update['user'])
        cols = np.array(df_update['item'])  # update_users
        Uupdate = sps.csr_matrix((vals, (rows, cols)), shape=(d.N + 1, d.M))
        data_update = MFAsym_data_iter_preset(df_update, Uupdate, rnorm=rnorm, Bi=Bi, remove_rated_items=False)
        envc.update_user(data_update)

        # check outputs match
        asvdc_rec_N = envc.recommend([d.N])
        self.assertTrue(np.equal(asvd_rec_0[0], asvdc_rec_N[0]).all())  # item scores
        self.assertTrue(np.allclose(asvd_rec_0[1], asvdc_rec_N[1]))  # item scores

if __name__=="__main__":
    unittest.main()