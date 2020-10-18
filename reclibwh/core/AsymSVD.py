import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
import tensorflow as tf
import parse
import scipy.sparse as sps

from tensorflow.keras import Model
from datetime import datetime
from ..utils.ItemMetadata import ExplicitDataFromCSV

from .Models import STANDARD_KERAS_SAVE_FMT, initialize_from_json
from .EvalProto import EvalProto, EvalCallback, AUCEval
from .Environment import Algorithm, UpdateAlgo, Environment
from .RecAlgos import MFAsymRecAlgo, SimpleMFRecAlgo
from .MatrixFactor import AsymSVDDataIterator, KerasModelSGD
from ..data.PresetIterators import MFAsym_data_iter_preset, AUC_data_iter_preset
from ..utils.utils import get_pos_ratings_padded, mean_nnz
from typing import Iterable, Optional
from tqdm import tqdm

class AsymSVDAlgo(KerasModelSGD):

    def predict(self, u_in=None, i_in=None, uj_in=None, bj_in=None, rj_in=None):
        env = self._KerasModelSGD__env
        self._KerasModelSGD__initialize()
        model = env.get_state()['model']
        return model.predict({'u_in': u_in, 'i_in': i_in, 'uj_in': uj_in, 'bj_in': bj_in, 'ruj_in': rj_in}, batch_size=5000)

# class AsymSVDCached(MatrixFactorizer):
#     """[
#         A cached version the asymmetric SVD, storing the X factors in one model
#         (model_X), and the derived user factors in another (model_main). When
#         an update happens (user adds/changes ratings), model_X is run to compute
#         updated user factors, and this is placed into model_main in the allocated
#         user slot.
#
#         Prediction is run on the main model which is much cheaper to run as it
#         does not require summing up user factors for every prediction made, which
#         can be very computationally intensive, esp for users with many ratings.
#
#         This model is non-trainable, as it is difficult to do so. So it is only
#         used for inference after an AsymSVD model is trained.
#     ]
#     """
#
#     def __init__(self, *args, **kwargs):
#         super(AsymSVDCached, self).__init__(*args, **kwargs)
#         self.model_X, self.model_conf_X = self.models[0], self.model_confs[0]
#         self.model_main, self.model_conf_main = self.models[1], self.model_confs[1]
#
#     def save(self, model_path='model.h5'):
#         self.model_X.save(os.path.join(self.model_path, model_path.rstrip('.h5')+'_X.h5'))
#         self.model_main.save(os.path.join(self.model_path, model_path.rstrip('.h5')+'_main.h5'))
#
#     def predict_X(self, ui, bj, ruj):
#         return self.model_X.predict({'ui_in': ui, 'bj_in': bj, 'ruj_in': ruj}, batch_size=5000)
#
#     def predict(self, u, i):
#         return self.model_main.predict({'u_in': u, 'i_in': i}, batch_size=5000)
#
#     def import_ASVD_weights(self, asvd, Utrain, bi, batchsize=1000):
#
#         print("importing ASVD weights")
#
#         N = Utrain.shape[0]
#         X = asvd.model.get_layer('Q').get_weights()
#         self.model_X.get_layer('X').set_weights(X)
#         P = asvd.model.get_layer('P').get_weights()
#         self.model_main.get_layer('P').set_weights(P)
#         Bi = asvd.model.get_layer('Bi').get_weights()
#         self.model_main.get_layer('Bi').set_weights(Bi)
#         Q = np.zeros(shape=(N,1,20))
#
#         users = np.arange(N)
#         batchsize=1000
#
#         for i in tqdm(range(0, len(users)//batchsize + 1)):
#             start = i*batchsize
#             stop = min((i+1)*batchsize, len(users))
#             u = users[start:stop]
#             self.update_users(Utrain, bi, u)
#
#     def _add_users(self, num=1):
#
#         self.data_conf['N'] += num
#
#         [_, new_model_main], [_, new_model_conf_main], _ = MatrixFactorizer.intialize_from_json(self.model_path, self.data_conf, saved_model=None, config_path=self.config_path)
#
#         # update old embeddings
#         oldX = self.model_main.get_layer('Q').get_weights()[0]
#         f = oldX.shape[1]
#         newX = np.concatenate([oldX, np.random.normal(0,1.,size=(num, f))])
#         new_model_main.get_layer('Q').set_weights([newX])
#         new_model_main.get_layer('P').set_weights(self.model_main.get_layer('P').get_weights())
#         new_model_main.get_layer('Bi').set_weights(self.model_main.get_layer('Bi').get_weights())
#         self.model_main = new_model_main
#         self.model_conf_main = new_model_conf_main
#         self.model_main.summary()
#         pass
#
#     def update_users(self, Utrain, bi, users: Iterable[int]):
#         rs, ys = get_pos_ratings_padded(Utrain, users, padding_val=0, offset_yp=1)
#         bjs = bi[ys-1]
#         Qnew = self.predict_X(ys, bjs, rs)
#         Q = self.model_main.get_layer('Q').get_weights()[0]
#         Q[users] = np.squeeze(Qnew)
#         self.model_main.get_layer('Q').set_weights([np.squeeze(Q)])
#
#         return
#
#     def evaluate(self, data: ExplicitDataFromCSV, users: Optional[Iterable[int]]=None):
#
#         _, (u_test, i_test, r_test) = data.make_training_datasets(dtype='dense', users=users)
#         if len(u_test) == 0: return
#         self.model_main.evaluate(x={'u_in': u_test, 'i_in': i_test}, y={'rhat': r_test}, batch_size=5000)

class AsymSVDCachedUpdater(UpdateAlgo):

    def __init__(self, env: Environment):
        super(AsymSVDCachedUpdater, self).__init__()
        self.__model_main = None
        self.__model_X = None
        self.__env = env

    def initialize(self):
        self.__model_main, self.__model_X = self.__env['model']

    def _predict_X(self, ui, bj, ruj):
        self.initialize()
        return self.__model_X.predict({'ui_in': ui, 'bj_in': bj, 'ruj_in': ruj}, batch_size=5000)

    def update_user(self, data):
        self.initialize()
        for us, rs, ys, bjs in data:
            Qnew = self._predict_X(ys, bjs, rs)
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
        self.__model_main, self.__model_X = self.__env['model']

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
        extra_callbacks += [EvalCallback(self, "eval.csv", self)]
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
            self, path, model, data, state, extra_callbacks=None, med_score=3.0
    ):
        Environment.__init__(self, path, model, data, state)
        if extra_callbacks is None: extra_callbacks = []
        extra_callbacks += [EvalCallback(self, "eval.csv", self)]
        AsymSVDCachedUpdater.__init__(self, self)
        AsymSVDCachedAlgo.__init__(self, self, self) # <--- this is silly, fix!
        SimpleMFRecAlgo.__init__(self, self, self, output_key=0)
        AUCEval.__init__(self, self, self, med_score)

if __name__=="__main__":

    data_folder = '/home/ong/personal/recommender/data/ml-latest-small-2'
    d = ExplicitDataFromCSV(True, data_folder=data_folder)
    df_train = d.get_ratings_split(0)
    df_test = d.get_ratings_split(1)

    Utrain, Utest = d.make_training_datasets(dtype='sparse')

    rnorm = {'loc': 0.0, 'scale': 5.0}
    data_train = MFAsym_data_iter_preset(df_test, Utest, rnorm=rnorm)
    auc_test_data = AUC_data_iter_preset(Utest)
    auc_train_data = AUC_data_iter_preset(Utrain)

    data = {
        "train_data": data_train,
        "valid_data": data_train,
        "auc_data": {'test': auc_test_data, 'train': auc_train_data},
        "mf_asym_rec_data": {'U': Utrain, 'norm': rnorm},
    }
    env_vars = {"save_fmt": STANDARD_KERAS_SAVE_FMT, "data_conf": {"M": d.M, "N": d.N}}
    m = initialize_from_json(data_conf={"M": d.M, "N": d.N}, config_path="SVD.json.template")[0]

    model_folder = '/home/ong/personal/recommender/models/test'
    save_path = os.path.join(model_folder, "MF_{:s}".format(datetime.now().strftime("%Y-%m-%d.%H-%M-%S")))
    if not os.path.exists(save_path): os.mkdir(save_path)

    model = AsymSVDEnv(save_path, m, data, env_vars, epochs=5, med_score=3.0)

    model.fit()