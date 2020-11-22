import unittest
from reclibwh.utils.ItemMetadata import ExplicitDataDummy
from reclibwh.core.Models import STANDARD_KERAS_SAVE_FMT, initialize_from_json
import numpy as np
from reclibwh.core.ALS import ALSEnv, ALSRef, run_als_step
import scipy.sparse as sps
from reclibwh.data.PresetIterators import ALS_data_iter_preset, AUC_data_iter_preset
from reclibwh.utils.testutils import refresh_dir
import tensorflow as tf

class TestALS(unittest.TestCase):

    def setUp(self) -> None:

        self.data = ExplicitDataDummy(3, 4, 1)
        self.save_path = "tests/models/test_ALS"

    def test_als_run_single_step(self):

        d = self.data
        U, _ = d.make_training_datasets(dtype='sparse')
        N = d.N
        M = d.M
        K = 5
        alpha = 1.
        lamb = 1e-02
        # TODO: very interesting, when lamb is too small, the L term in ALS seems to have a humungous condition number
        # causing the estimates of Xnew in numpy and tensorflow to be vastly different, perhaps due to a difference
        # in the algorithm used?

        Xinit = np.random.normal(0, 1 / np.sqrt(K), size=(N, K))
        Yinit = np.random.normal(0, 1 / np.sqrt(K), size=(M, K))


        p = U.copy().astype(np.bool).astype(np.float, copy=True)
        C = U.copy()
        C.data = 1 + alpha * C.data
        als = ALSRef(N=N, M=M, K=K, lamb=lamb, alpha=alpha)
        Xnew = als._run_single_step(Yinit, Xinit, C, U, p)

        data = ALS_data_iter_preset(U, batchsize=N)
        X = tf.Variable(Xinit, dtype=tf.float32)
        Y = tf.Variable(Yinit, dtype=tf.float32)
        Y2 = tf.reshape(tf.tensordot(tf.transpose(Y), Y, axes=1), shape=[K, K])
        LambdaI = lamb * tf.eye(K)

        for x, y in data:
            us = tf.convert_to_tensor(x['u_in'], dtype=tf.int32)
            ys = tf.convert_to_tensor(x['i_in'], dtype=tf.int32)
            rs = tf.convert_to_tensor(y['rhat'], dtype=tf.float32)
            diff = run_als_step(X, Y, Y2, LambdaI, us, ys, rs, alpha)

        print(X.numpy(), Xnew)
        # self.assertTrue(np.allclose(X.numpy(), Xnew))

    def test_als(self):

        d = self.data
        Utrain, Utest = d.make_training_datasets(dtype='sparse')

        M = d.M + 1
        N = d.N + 1

        save_path = self.save_path
        refresh_dir(save_path) # deletes everything inside save_path

        train_data_X = ALS_data_iter_preset(Utrain, batchsize=20)
        train_data_Y = ALS_data_iter_preset(sps.csr_matrix(Utrain.T), batchsize=20)
        auc_test_data = AUC_data_iter_preset(Utest)
        auc_train_data = AUC_data_iter_preset(Utrain)

        data = {
            "train_data_X": train_data_X, "train_data_Y": train_data_Y,
            "auc_data": {'test': auc_test_data, 'train': auc_train_data}
        }

        env_vars = {
            "save_fmt": STANDARD_KERAS_SAVE_FMT,
            "data_conf": {"M": M, "N": N},
            "data": data
        }

        m = initialize_from_json(data_conf={"M": M + 1, "N": N + 1}, config_path="ALS.json.template")
        alsenv = ALSEnv(save_path, m, data, env_vars, epochs=3)
        alsenv.fit()
        auc = alsenv.evaluate()
        self.assertLess(0.9, auc) # auc > 0.9

        # update first and last user
        rows, cols, vals = sps.find(Utrain[0])
        rows = np.concatenate([rows, np.ones(dtype=int, shape=rows.shape) * (N - 1)])
        cols = np.concatenate([cols, cols])
        vals = np.concatenate([vals, vals])
        Uupdate = sps.csr_matrix((vals, (rows, cols)), shape=(N, M))
        update_data = ALS_data_iter_preset(Uupdate, rows=[0, N - 1])

        alsenv.update_user(update_data)

        self.assertTrue(np.allclose(
            alsenv.get_state()['model'][0].get_layer('X').get_weights()[0][0],
            alsenv.get_state()['model'][0].get_layer('X').get_weights()[0][N - 1]
        ))

if __name__=="__main__":
    unittest.main()