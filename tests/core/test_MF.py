import unittest
from reclibwh.utils.ItemMetadata import ExplicitDataDummy
from reclibwh.core.Models import STANDARD_KERAS_SAVE_FMT, initialize_from_json
import os
import numpy as np
from reclibwh.core.MatrixFactor import MatrixFactorizerEnv
from reclibwh.data.PresetIterators import MF_data_iter_preset, AUC_data_iter_preset
from reclibwh.utils.test_utils import refresh_dir

class TestLMF(unittest.TestCase):

    def setUp(self) -> None:

        self.data = ExplicitDataDummy()
        self.save_path = "tests/models/test_MF"

    def test_lmf(self):

        d = self.data
        df_train , df_test = d.make_training_datasets(dtype='df')
        Utrain, Utest = d.make_training_datasets(dtype='sparse')
        M = d.M
        N = d.N

        rnorm = {'loc': 0.0, 'scale': 5.0}
        mfit = MF_data_iter_preset(df_train, rnorm=rnorm)
        auc_test_data = AUC_data_iter_preset(Utest)
        auc_train_data = AUC_data_iter_preset(Utrain)

        data = {"train_data": mfit, "valid_data": mfit, "auc_data": {'test': auc_test_data, 'train': auc_train_data}}
        env_vars = {"save_fmt": STANDARD_KERAS_SAVE_FMT, "data_conf": {"M": M, "N": N}}
        m = initialize_from_json(data_conf={"M": M, "N": N}, config_path="SVD.json.template")[0]

        save_path = os.path.join(self.save_path, 'test_asym_svd_train')
        refresh_dir(save_path)

        mf = MatrixFactorizerEnv(save_path, m, data, env_vars, epochs=5, med_score=3.0)
        res = mf.fit()
        self.assertLess(res[2], 0.01)

if __name__=="__main__":
    unittest.main()