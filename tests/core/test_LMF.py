import unittest
from reclibwh.utils.ItemMetadata import ExplicitDataDummy
from reclibwh.core.Models import STANDARD_KERAS_SAVE_FMT, initialize_from_json
import os
import numpy as np
from reclibwh.core.LMF import LMFEnv
from reclibwh.data.PresetIterators import LMF_data_iter_preset, AUC_data_iter_preset
from reclibwh.utils.test_utils import refresh_dir

class TestLMF(unittest.TestCase):

    def setUp(self) -> None:

        self.data = ExplicitDataDummy()
        self.save_path = "tests/models/test_LMF"

    def test_lmf(self):

        d = self.data
        Utrain, Utest = d.make_training_datasets(dtype='sparse')

        train_mfit = LMF_data_iter_preset(Utrain)
        auc_test_data = AUC_data_iter_preset(Utest)
        auc_train_data = AUC_data_iter_preset(Utrain)

        data = {
            "train_data": train_mfit,
            "auc_data": {'test': auc_test_data, 'train': auc_train_data},
            "lmfc_data": {'test': Utrain, 'train': Utest}
        }

        env_vars = {
            "save_fmt": STANDARD_KERAS_SAVE_FMT,
            "data_conf": {"M": d.M, "N": d.N},
        }

        m = initialize_from_json(data_conf={"M": d.M, "N": d.N}, config_path="SVD.json.template")[0]

        save_path = os.path.join(self.save_path, 'test_asym_svd_train')
        refresh_dir(save_path)

        lmf = LMFEnv(save_path, m, data, env_vars, epochs=3)
        lmf.fit()
        auc = lmf.evaluate()
        self.assertLess(0.9, auc)

if __name__=="__main__":
    unittest.main()