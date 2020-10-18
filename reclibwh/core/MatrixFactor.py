import numpy as np
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, LambdaCallback
import tensorflow as tf
from ..utils.utils import get_pos_ratings_padded, mean_nnz
import scipy.sparse as sps

from datetime import datetime
from ..utils.ItemMetadata import ExplicitDataFromCSV
import json

from .Environment import Environment, Algorithm
from .Models import STANDARD_KERAS_SAVE_FMT, initialize_from_json, model_restore
from ..data.iterators import EpochIterator, BasicDFDataIter, Normalizer, Rename, XyDataIterator,  SparseMatRowIterator
from .EvalProto import EvalProto, EvalCallback, AUCEval
from .RecAlgos import SimpleMFRecAlgo
import pandas as pd
from ..data.PresetIterators import MF_data_iter_preset, AUC_data_iter_preset

class KerasModelSGD(Algorithm):

    def __init__(self, env: Environment, epochs=30, early_stopping=True, tensorboard=True, extra_callbacks=None):
        config = {}
        config['epochs'] = epochs
        config['early_stopping'] = early_stopping
        config['tensorboard'] = tensorboard
        config['start_epoch'] = 0
        self.__extra_callbacks = extra_callbacks
        self.__initialized = False
        self.__env = env
        self.__config = config

    def __save(self):
        env = self.__env
        path = env.get_state()['environment_path']
        with open(os.path.join(path, 'algorithm_config.json'), 'w') as fp: json.dump(self.__config, fp)

    def __save_config_cb(self, path, epoch):
        self.__config['start_epoch'] = epoch
        with open(os.path.join(path, 'algorithm_config.json'), 'w') as fp: json.dump(self.__config, fp)
    
    def __env_restore(self):
        state = self.__env.get_state()
        path = state['environment_path']
        if not os.path.exists(os.path.join(path, 'algorithm_config.json')):
            print("algorithm config does not exist. cannot restore")
        else:
            with open(os.path.join(path, 'algorithm_config.json'), 'r') as fp: self.__config = json.load(fp)
        rest = model_restore(environment_path=path, saved_model=state.get('use_model'), save_fmt=state.get('save_fmt', STANDARD_KERAS_SAVE_FMT))
        if rest:
            state['model'] = rest['model'][0]
            self.__config['start_epoch'] = rest['start_epoch']

    def __initialize(self):
        """
        Initialize the algorithm state by restoring config, model etc
        :param env:
        :return:
        """
        if not self.__initialized: self.__env_restore()
        self.__initialized = True
        return
    
    def fit(self):

        self.__initialize()
        state = self.__env.get_state()
        # mandatory 
        model = state['model']
        data = state['data']
        model_path = state['environment_path']
        # optional
        save_fmt = state.get('save_fmt', STANDARD_KERAS_SAVE_FMT)

        start_epoch = self.__config['start_epoch']
        tensorboard = self.__config['tensorboard']
        epochs = self.__config['epochs']
        early_stopping = self.__config['early_stopping']
        extra_callbacks = self.__extra_callbacks

        train_data = data['train_data']
        valid_data = data['valid_data']

        steps_per_epoch = len(train_data)
        validation_batch_size = 1
        validation_steps = len(valid_data)

        if tensorboard:
            tensorboard_path = os.path.join(model_path, 'tensorboard_logs')
            file_writer = tf.summary.create_file_writer(tensorboard_path + "/metrics")
            file_writer.set_as_default()
        
        callbacks = [
            ModelCheckpoint(os.path.join(model_path, save_fmt)),
            CSVLogger(os.path.join(model_path, 'history.csv')),
            LambdaCallback(on_epoch_end=lambda epoch, logs: self.__save_config_cb(model_path, epoch))
        ] + \
        ([EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)] if early_stopping else []) + \
        ([TensorBoard(log_dir=tensorboard_path)] if tensorboard else []) + \
        (extra_callbacks if extra_callbacks is not None else [])

        model.fit(
            EpochIterator(epochs)(train_data).__iter__(),  steps_per_epoch=steps_per_epoch, epochs=epochs,
            validation_data=EpochIterator(epochs)(valid_data).__iter__(), 
            validation_batch_size=validation_batch_size, validation_steps=validation_steps, 
            verbose=1, initial_epoch=start_epoch,
            shuffle=True,
            callbacks=callbacks
        )
        self.__env.set_state({'model': model})
        return

    def predict(self, u_in=None, i_in=None, **kwargs):
        self.__initialize()
        model = self.__env['model']
        return model.predict({'u_in': u_in, 'i_in': i_in}, batch_size=5000)

    def __evaluate(self, data=None):
        self.__initialize()
        model = self.__env['model']
        return model.evaluate(EpochIterator(1)(data).__iter__(), batch_size=5000)

def transplant_weights(model_from: Model, model_to: Model):

    """
    Tries to move the weights from one model to another, with the same layers (and layer types)
    but possibly different weight dimensions. It is assumed both layers have same number of weight tensors,
    and the tensors are of the same dimensionality.
    For each pair of weight tensors, a common hypercube is cut out from the model model_from and copied
    into the same common hypercude in the model model_to
    :param model_from:
    :param model_to:
    :return:
    """
    
    assert len(model_from.layers)==len(model_to.layers)

    for layer_from, layer_to in zip(model_from.layers, model_to.layers):
    
        print("transplanting layer {:s} -> {:s} ".format(layer_from.name, layer_to.name))

        weights_from = layer_from.get_weights()
        weights_to = layer_to.get_weights()
        assert len(weights_from) == len(weights_to)
        for wf, wt in zip(weights_from, weights_to):
            assert len(wf.shape) == len(wt.shape)
            print("    transplanting weights {} -> {} ".format(wf.shape, wt.shape))
            common_hypercube = tuple([slice(0,min(sf, st)) for sf, st in zip(wf.shape, wt.shape)])
            wt[tuple(common_hypercube)] = wf[tuple(common_hypercube)]
        layer_to.set_weights(weights_to)

class MatrixFactorizerEnv(Environment, KerasModelSGD, SimpleMFRecAlgo, AUCEval):

    def __init__(
            self, path, model, data, state,
            epochs=30, early_stopping=True, tensorboard=True, extra_callbacks=None,
            med_score=3.0
    ):
        Environment.__init__(self, path, model, data, state)
        if extra_callbacks is None: extra_callbacks = []
        extra_callbacks += [EvalCallback(self, "eval.csv", self)]
        KerasModelSGD.__init__(self, self, epochs, early_stopping, tensorboard, extra_callbacks)
        SimpleMFRecAlgo.__init__(self, self, self, output_key=0)
        AUCEval.__init__(self, self, self, med_score)

if __name__=="__main__":

    data_folder = '/home/ong/personal/recommender/data/ml-latest-small-2'
    d = ExplicitDataFromCSV(True, data_folder=data_folder)
    df_train = d.get_ratings_split(0)
    df_test = d.get_ratings_split(1)

    Utrain, Utest = d.make_training_datasets(dtype='sparse')

    # auc_test_data = SparseMatRowIterator(10, padded=True, negative=False)(
    #     {'S': Utrain, 'pad_val': -1., 'rows': np.arange(0, d.N, d.N // 300)})
    # auc_train_data = SparseMatRowIterator(10, padded=True, negative=False)(
    #     {'S': Utest, 'pad_val': -1., 'rows': np.arange(0, d.N, d.N // 300)})
    auc_test_data = AUC_data_iter_preset(Utest)
    auc_train_data = AUC_data_iter_preset(Utrain)

    # df = pd.DataFrame({
    #     'user': [1,2,3,4],
    #     'item': [1,2,3,4],
    #     'rating': [1.,2.,3.,4.],
    # })
    rnorm = {'loc': 0.0, 'scale': 5.0}
    # U = sps.csr_matrix((df['rating'], (df['user'], df['item'])))

    # TODO: wrap in a factory function or something
    # it = BasicDFDataIter(200)([df_train])
    # nit = Normalizer({'rating': rnorm})(it)
    # rename = Rename({'user': 'u_in', 'item': 'i_in', 'rating': 'rhat'})(nit)
    # mfit = XyDataIterator(ykey='rhat')(rename)
    mfit = MF_data_iter_preset(df_train, rnorm=rnorm)

    data = {"train_data": mfit, "valid_data": mfit, "auc_data": {'test': auc_test_data, 'train': auc_train_data}}
    env_vars = {"save_fmt": STANDARD_KERAS_SAVE_FMT, "data_conf": {"M": d.M, "N": d.N}}
    m = initialize_from_json(data_conf={"M": d.M, "N": d.N}, config_path="SVD.json.template")[0]
    # algo = KerasModelSGD(epochs=5)
    # rec = SimpleMFRecAlgo(output_key=0)

    model_folder = '/home/ong/personal/recommender/models/test'
    save_path = os.path.join(model_folder, "MF_{:s}".format(datetime.now().strftime("%Y-%m-%d.%H-%M-%S")))
    if not os.path.exists(save_path): os.mkdir(save_path)
    # env = Environment(save_path, model=m, data=data, algo=algo, state=env_vars, rec=rec)

    mfenv = MatrixFactorizerEnv(save_path, m, data, env_vars, epochs=5, med_score=3.0)
    mfenv.fit()