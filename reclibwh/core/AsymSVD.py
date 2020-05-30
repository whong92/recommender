import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
import tensorflow as tf
import parse
import scipy.sparse as sps

from datetime import datetime
from ..utils.ItemMetadata import ExplicitDataFromCSV

from .MatrixFactor import parse_config_json, get_model_config, make_model, MatrixFactorizer, AsymSVDDataIterator
from ..utils.utils import get_pos_ratings_padded, mean_nnz
from typing import Iterable, Optional
from tqdm import tqdm

class AsymSVD(MatrixFactorizer):
    
    def make_data_iterator(self, data: ExplicitDataFromCSV, train=True):
        batchsize = self.model_conf['batchsize'] if train else 5000
        epochs = self.model_conf['epochs']
        return AsymSVDDataIterator(data, batchsize, epochs, train=train)

    def predict(self, u, i, uj, bj, rj):
        return self.model.predict({'u_in': u, 'i_in': i, 'uj_in': uj, 'bj_in': bj, 'ruj_in': rj}, batch_size=5000)

class AsymSVDCached:
    """[
        A cached version the asymmetric SVD, storing the X factors in one model
        (model_X), and the derived user factors in another (model_main). When
        an update happens (user adds/changes ratings), model_X is run to compute
        updated user factors, and this is placed into model_main in the allocated
        user slot.

        Prediction is run on the main model which is much cheaper to run as it
        does not require summing up user factors for every prediction made, which
        can be very computationally intensive, esp for users with many ratings.

        This model is non-trainable, as it is difficult to do so. So it is only
        used for inference after an AsymSVD model is trained.
    ]
    """

    def __init__(
        self, model_dir, N, M, Nranked, mode='train', config_path_X=None, config_path_main=None, saved_model=None
    ):
        data_conf = {'N': N, 'M': M}
        self.model_path = model_dir
        
        self.config_path_X = config_path_X
        self.config_path_main = config_path_main
        self.data_conf = data_conf

        saved_model_X = saved_model.rstrip(".h5")+"_X.h5" if saved_model is not None else None
        saved_model_main = saved_model.rstrip(".h5")+"_main.h5" if saved_model is not None else None

        self.model_X, self.model_conf_X, _ = MatrixFactorizer.intialize_from_json(model_dir, data_conf, saved_model=saved_model_X,  config_path=config_path_X)

        self.model_main, self.model_conf_main, _ = MatrixFactorizer.intialize_from_json(model_dir, data_conf, saved_model=saved_model_main,  config_path=config_path_main)
    
    def save(self, model_path='model.h5'):
        self.model_X.save(os.path.join(self.model_path, model_path.rstrip('.h5')+'_X.h5'))
        self.model_main.save(os.path.join(self.model_path, model_path.rstrip('.h5')+'_main.h5'))
    
    def predict_X(self, ui, bj, ruj):
        return self.model_X.predict({'ui_in': ui, 'bj_in': bj, 'ruj_in': ruj}, batch_size=5000)

    def predict(self, u, i):
        return self.model_main.predict({'u_in': u, 'i_in': i}, batch_size=5000)
    
    def import_ASVD_weights(self, asvd, Utrain, bi, batchsize=1000):

        print("importing ASVD weights")

        N = Utrain.shape[0]
        X = asvd.model.get_layer('Q').get_weights()
        self.model_X.get_layer('X').set_weights(X)
        P = asvd.model.get_layer('P').get_weights()
        self.model_main.get_layer('P').set_weights(P)
        Bi = asvd.model.get_layer('Bi').get_weights()
        self.model_main.get_layer('Bi').set_weights(Bi)
        Q = np.zeros(shape=(N,1,20))

        users = np.arange(N)
        batchsize=1000

        for i in tqdm(range(0, len(users)//batchsize + 1)):
            start = i*batchsize
            stop = min((i+1)*batchsize, len(users))
            u = users[start:stop]
            self.update_users(Utrain, bi, u)

    def _add_users(self, num=1):

        self.data_conf['N'] += num

        new_model_main, new_model_conf_main, _ = MatrixFactorizer.intialize_from_json(self.model_path, self.data_conf, saved_model=None, config_path=self.config_path_main)

        # update old embeddings
        oldX = self.model_main.get_layer('Q').get_weights()[0]
        f = oldX.shape[1]
        newX = np.concatenate([oldX, np.random.normal(0,1.,size=(num, f))])
        new_model_main.get_layer('Q').set_weights([newX])
        new_model_main.get_layer('P').set_weights(self.model_main.get_layer('P').get_weights())
        new_model_main.get_layer('Bi').set_weights(self.model_main.get_layer('Bi').get_weights())
        self.model_main = new_model_main
        self.model_conf_main = new_model_conf_main
        self.model_main.summary()
        pass
    
    def update_users(self, Utrain, bi, users: Iterable[int]):
        rs, ys = get_pos_ratings_padded(Utrain, users, padding_val=0, offset_yp=1)
        bjs = bi[ys-1]
        Qnew = self.predict_X(ys, bjs, rs)
        Q = self.model_main.get_layer('Q').get_weights()[0]
        Q[users] = np.squeeze(Qnew)
        self.model_main.get_layer('Q').set_weights([np.squeeze(Q)])
        
        return

    def evaluate(self, data: ExplicitDataFromCSV, users: Optional[Iterable[int]]=None):

        _, (u_test, i_test, r_test) = data.make_training_datasets(dtype='dense', users=users)
        self.model_main.evaluate(x={'u_in': u_test, 'i_in': i_test}, y={'rhat': r_test}, batch_size=5000)

if __name__=="__main__":

    data_folder = '/home/ong/personal/recommender/data/ml-latest-small-2'
    model_folder = '/home/ong/personal/recommender/models/MF_tmp'
    save_path = os.path.join(model_folder, "MF_{:s}".format(datetime.now().strftime("%Y-%m-%d.%H-%M-%S")))
    d = ExplicitDataFromCSV(True, data_folder=data_folder)

    data_train, data_test = d.make_training_datasets(dtype='dense')

    model = AsymSVD(
        model_dir=model_folder, N=d.N, M=d.M, Nranked=d.Nranked, 
        config_path='/home/ong/personal/recommender/reclibwh/core/model_templates/SVD_asym.json.template'
    )

    model.fit(d)