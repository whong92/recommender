import numpy as np
import os
import scipy.sparse as sps

from tensorflow.keras import Model
from datetime import datetime
from ..utils.ItemMetadata import ExplicitDataFromCSV, ExplicitDataDummy

from .Models import STANDARD_KERAS_SAVE_FMT, initialize_from_json
from .EvalProto import EvalProto, EvalCallback, AUCEval
from .Environment import Algorithm, UpdateAlgo, Environment
from .RecAlgos import MFAsymRecAlgo, SimpleMFRecAlgo
from .MatrixFactor import KerasModelSGD
from ..data.PresetIterators import MFAsym_data_iter_preset, AUC_data_iter_preset, MFAsym_data_tf_dataset
import pandas as pd
import argparse

class AsymSVDAlgo(KerasModelSGD):

    def predict(self, u_in=None, i_in=None, uj_in=None, bj_in=None, rj_in=None):
        env = self._KerasModelSGD__env
        self._KerasModelSGD__initialize()
        model = env.get_state()['model']
        return model.predict({'u_in': u_in, 'i_in': i_in, 'uj_in': uj_in, 'bj_in': bj_in, 'ruj_in': rj_in}, batch_size=5000)

class AsymSVDCachedUpdater(UpdateAlgo):

    def __init__(self, env: Environment):
        UpdateAlgo.__init__(self)
        self.__model_main = None
        self.__model_X = None
        self.__env = env

    def __initialize(self):
        self.__model_X, self.__model_main = self.__env['model']

    def _predict_X(self, ui, bj, ruj):
        self.__initialize()
        return self.__model_X.predict({'ui_in': ui, 'bj_in': bj, 'ruj_in': ruj}, batch_size=5000)

    def update_user(self, data):

        self.__initialize()

        for d in data:

            X, y = d
            us = X['u_in']
            rs = X['ruj_in']
            bjs = X['bj_in']
            ui = X['uj_in']

            Qnew = self._predict_X(ui, bjs, rs)
            Q = self.__model_main.get_layer('Q').get_weights()[0]
            Q[us] = np.squeeze(Qnew)
            self.__model_main.get_layer('Q').set_weights([np.squeeze(Q)])

class AsymSVDCachedAlgo(Algorithm):

    def __init__(self, env: Environment, updater: AsymSVDCachedUpdater):
        """
        Dummy implementation for now
        """
        self.__env = env
        self.__model_main = None
        self.__model_X = None
        self.__updater = updater

    def initialize(self):
        self.__model_X, self.__model_main = self.__env['model']

    def __env_save(self):
        # TODO
        pass

    def __env_restore(self):
        # TODO
        pass

    def predict(self, u_in=None, i_in=None):
        self.initialize()
        return self.__model_main.predict({'u_in': u_in, 'i_in': i_in}, batch_size=5000)

    def _import_ASVD_weights(self, asvd: Model, data):
        self.initialize()
        print("importing ASVD weights")

        X = asvd.get_layer('Q').get_weights()
        self.__model_X.get_layer('X').set_weights(X)
        P = asvd.get_layer('P').get_weights()
        self.__model_main.get_layer('P').set_weights(P)
        Bi = asvd.get_layer('Bi').get_weights()
        self.__model_main.get_layer('Bi').set_weights(Bi)

        self.__updater.update_user(data)

    def fit(self):
        print("fiting by importing weights from asvd")
        asvd = self.__env['data']['asvd']
        train_data = self.__env['data']['train_data']
        self._import_ASVD_weights(asvd, train_data)

class AsymSVDEnv(Environment, AsymSVDAlgo, MFAsymRecAlgo, AUCEval):

    def __init__(
            self, path, model, data, state,
            epochs=30, early_stopping=True, tensorboard=False, extra_callbacks=None, med_score=3.0
    ):
        Environment.__init__(self, path, model, data, state)
        if extra_callbacks is None: extra_callbacks = []
        # extra_callbacks += [EvalCallback(self, "eval.csv", self)]
        AsymSVDAlgo.__init__(self, self, epochs, early_stopping, tensorboard, extra_callbacks)
        MFAsymRecAlgo.__init__(self, self, self, output_key=0)
        AUCEval.__init__(self, self, self, med_score)


class AsymSVDCachedEnv(
    Environment,
    AsymSVDCachedAlgo,
    AsymSVDCachedUpdater,
    SimpleMFRecAlgo,
    AUCEval
):

    def __init__(
        self, path, model, data, state, med_score=3.0
    ):
        Environment.__init__(self, path, model, data, state)
        AsymSVDCachedUpdater.__init__(self, self)
        AsymSVDCachedAlgo.__init__(self, self, self) # <--- this is silly, fix!
        SimpleMFRecAlgo.__init__(self, self, self, output_key=0)
        AUCEval.__init__(self, self, self, med_score)

if __name__=="__main__":


    now_str = datetime.now().strftime("%Y-%m-%d.%H-%M-%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", "-d", type=str, default="data/ml-latest-small")
    parser.add_argument("--asvd_model_folder", "-asvd", type=str, default="models/ASVD_{:s}".format(now_str))
    parser.add_argument("--asvdc_model_folder", "-asvdc", type=str, default="models/ASVDC_{:s}".format(now_str))
    args = parser.parse_args()
    data_folder = args.data_folder
    save_path_asvd = args.asvd_model_folder
    save_path_asvdc = args.asvdc_model_folder

    d = ExplicitDataFromCSV(True, data_folder=data_folder)
    # d = ExplicitDataDummy()
    df_train, df_test = d.make_training_datasets(dtype='df')
    Utrain, Utest = d.make_training_datasets(dtype='sparse')
    M = d.M
    N = d.N
    Nranked = d.Nranked

    data_conf = {"M": M, "N": N, "Nranked": Nranked}

    rnorm = {'loc': 0.0, 'scale': 5.0}
    data_train = MFAsym_data_tf_dataset(df_train, Utrain, rnorm=rnorm, batchsize=2000)
    data_test = MFAsym_data_tf_dataset(df_test, Utrain, rnorm=rnorm, batchsize=2000)

    auc_test_data = AUC_data_iter_preset(Utest, rows=np.arange(0, N, N//300))
    auc_train_data = AUC_data_iter_preset(Utrain, rows=np.arange(0, N, N//300))

    data = {
        "train_data": data_train,
        "valid_data": data_test,
        "auc_data": {'test': auc_test_data, 'train': auc_train_data},
        "mf_asym_rec_data": {'U': Utrain, 'norm': rnorm},
    }
    env_vars = {"save_fmt": STANDARD_KERAS_SAVE_FMT, "data_conf": data_conf}
    m = initialize_from_json(
        data_conf=data_conf, config_path="SVD_asym.json.template",
        config_override={"SVD_asym": {"lr": 0.005, "batchsize": 2000}}
    )[0]

    if not os.path.exists(save_path_asvd): os.mkdir(save_path_asvd)

    env = AsymSVDEnv(save_path_asvd, m, data, env_vars, epochs=30, med_score=3.0)
    r = env.fit()
    m = env['model']

    if not os.path.exists(save_path_asvdc): os.mkdir(save_path_asvdc)
    mc = initialize_from_json(data_conf=data_conf, config_path="SVD_asym_cached.json.template")
    asvdc_data = MFAsym_data_iter_preset(
        pd.DataFrame({'user': np.arange(N), 'item':np.zeros(shape=(N,)), 'rating': np.zeros(shape=(N,))}),
        Utrain, rnorm=rnorm
    )
    env_varsc = {'data': {'asvd': m, 'train_data': asvdc_data}, 'data_conf': data_conf}
    envc = AsymSVDCachedEnv(save_path_asvdc, mc, data, env_varsc)
    envc.fit()
