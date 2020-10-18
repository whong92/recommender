import numpy as np
import os
import scipy.sparse as sps

from keras.utils import generic_utils
import tensorflow as tf
from typing import List
from .EvalProto import EvalProto, EvalCallback, AUCEval
import pandas as pd

from sklearn.metrics import mean_squared_error, average_precision_score
from reclibwh.utils.utils import get_pos_ratings, get_neg_ratings
import json

from datetime import datetime
from ..data.iterators import EpochIterator, Rename, XyDataIterator, SparseMatRowIterator, Normalizer
from .Environment import Environment, Algorithm
from .Models import STANDARD_KERAS_SAVE_FMT, initialize_from_json, model_restore
from .Losses import PLMFLoss
from .RecAlgos import SimpleMFRecAlgo
from ..data.PresetIterators import AUC_data_iter_preset, LMF_data_iter_preset

class LMFCEval(EvalProto):

    @property
    def __M(self):
        return self.__env.get_state()['data_conf']['M']

    @property
    def __N(self):
        return self.__env.get_state()['data_conf']['N']

    @property
    def __data(self):
        return self.__env.get_state()['data']['lmfc_data']

    def __init__(self, env: Environment, algo: Algorithm, users=None):
        self.__env = env
        # option to limit the number of users tested against, for faster testing
        self.__users = users
        self.__algo = algo

    def __mse_model(self, u_in, i_in, p):
        phat, _, _, _ = self.__algo.predict(u_in=u_in, i_in=i_in)
        return mean_squared_error(phat, p)

    def evaluate(self):

        users = self.__users if self.__users else np.arange(self.__N)
        M = self.__M

        Utest, U = self.__data['test'], self.__data['train']

        up, yp, rp = get_pos_ratings(Utest, users, M)
        uup, nup = np.unique(up, return_counts=True)
        rp = rp.astype(np.bool).astype(np.float, copy=True)
        un, yn = get_neg_ratings(U, users, M, samples_per_user=nup)
        rn = np.zeros(shape=un.shape, dtype=float)
        rn = rn.astype(np.bool).astype(np.float, copy=True)

        ep_mse = np.sqrt(self.__mse_model(up, yp, rp))
        en_mse = np.sqrt(self.__mse_model(un, yn, rn))

        u_test = np.concatenate([up, un])
        i_test = np.concatenate([yp, yn])
        r_test = np.concatenate([rp, rn]).astype(np.bool).astype(np.float, copy=True)
        phat, _, _, _ = self.__algo.predict(u_in=u_test, i_in=i_test)
        ap = average_precision_score(r_test, phat)

        print({'ap': ap, 'ep_mse': ep_mse, 'en_mse': en_mse})

        return ap


# @tf.function(experimental_relax_shapes=True)
def train_step(
        model: tf.keras.models.Model, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor,
        loss_fn: PLMFLoss, acc_grads: List[tf.Tensor], acc_loss: tf.Tensor):

    print("tracing tf function graph...")

    with tf.GradientTape() as tape:
        _, rhat, _, _ = model([x, y])
        loss_value = loss_fn(z, rhat)

    grads = tape.gradient(loss_value, model.trainable_weights)

    for acc_grad, grad in zip(acc_grads, grads):
        acc_grad.assign_add(grad)

    acc_loss.assign_add(loss_value)

    return loss_value, grads

def load_custom_checkpoint(ckpt, model, opt):

    tf_ckpt_path = ckpt['tf_ckpt_path']
    tf_ckpt_dir = ckpt['tf_ckpt_dir']

    tf_ckpt = tf.train.Checkpoint(optimizer=opt, model=model)
    tf_manager = tf.train.CheckpointManager(tf_ckpt, tf_ckpt_dir, max_to_keep=3)
    if tf_ckpt_path is None: tf_ckpt_path = tf_manager.latest_checkpoint
    tf_ckpt.restore(tf_ckpt_path)
    if tf_ckpt_path: print("Restored from {}".format(tf_manager.latest_checkpoint))
    else: print("Initializing from scratch.")

    return tf_ckpt, tf_manager

def save_ckpt(ckpt, tf_manager: tf.train.CheckpointManager):

    tf_ckpt_path = tf_manager.save()
    ckpt['tf_ckpt_path'] = tf_ckpt_path

def load_metrics(metrics_path):
    with open(metrics_path, 'r') as fp: return json.load(fp)

def save_metrics(metrics, metrics_path):
    with open(metrics_path, 'w') as fp: json.dump(metrics, fp)

class LMFGD(Algorithm):

    def __init__(self, env: Environment, epochs=30, batchsize=100, alpha=40, callbacks=None):

        config = {}
        config['epochs'] = epochs
        config['batchsize'] = batchsize
        config['alpha'] = alpha
        config['ckpt'] = {'epoch' :0, 'tf_ckpt_path': None, 'tf_ckpt_dir': None, 'metrics': None}

        self.__config = config
        self.__opt = tf.keras.optimizers.Adam(0.1)
        self.__tf_ckpt = None
        self.__tf_manager = None
        self.__trace = np.zeros(shape=(epochs * 2,))
        self.__initialized = False
        self.__env = env
        self.__callbacks = callbacks if callbacks is not None else []

    def __env_save(self):

        path = self.__env.get_state()['environment_path']
        ckpt = self.__config['ckpt']
        save_ckpt(ckpt, self.__tf_manager)
        save_metrics(list(self.__trace), ckpt['metrics']) # TODO: replace with tensorboard or something less unwieldy
        with open(os.path.join(path, 'algorithm_config.json'), 'w') as fp: json.dump(self.__config, fp)

    def __save_config_cb(self, epoch):
        self.__config['start_epoch'] = epoch
        self.__env_save()

    def __initialize(self):
        if not self.__initialized: self.__env_restore()
        self.__initialized = True

    def __env_restore(self):

        path = self.__env.get_state()['environment_path']
        model = self.__env.get_state()['model']
        ckpt = self.__config['ckpt']
        ckpt['tf_ckpt_dir'] = os.path.join(path, 'tf_ckpt')
        ckpt['metrics'] = os.path.join(path, 'metrics.json') # TODO: replace with tensorboard or something less unwieldy
        opt = self.__opt
        self.__tf_ckpt, self.__tf_manager = load_custom_checkpoint(ckpt, opt, model)
        if not os.path.exists(os.path.join(path, 'algorithm_config.json')):
            print("algorithm config does not exist. cannot restore")
            return
        with open(os.path.join(path, 'algorithm_config.json'), 'r') as fp: self.__config = json.load(fp)

    def __get_model_vars(self, model):
        vars = {'X': model.get_layer('P'), 'Y': model.get_layer('Q')}
        try:
            vars.update({'Bu': model.get_layer('Bu'), 'Bi': model.get_layer('Bi'), })
        except ValueError:
            print("no bias detected for model")
        return vars

    def fit(self):

        self.__initialize()
        state = self.__env.get_state()

        model = state['model']
        data = state['data']
        model_path = state['environment_path']
        callbacks = self.__callbacks

        alpha = self.__config['alpha']
        epochs = self.__config['epochs']
        batchsize = self.__config['batchsize']
        ckpt = self.__config['ckpt']
        opt = self.__opt
        start_epoch = ckpt['epoch']
        trace = self.__trace

        train_data = data['train_data']
        steps_per_epoch = len(train_data)
        epoch_train_data = EpochIterator((epochs-start_epoch)*2)(train_data)

        vars = self.__get_model_vars(model)

        X = vars['X']
        Y = vars['Y']
        Bu = vars.get('Bu')
        Bi = vars.get('Bi')
        loss_fn = PLMFLoss(alpha=alpha)

        num_seen = tf.Variable(initial_value=0., dtype='float32')
        cur_batchsize = tf.Variable(initial_value=0., dtype='float32')
        acc_loss = tf.Variable(initial_value=0.)

        phaseVariables = {
            'X': {'vars': [X, Bu], 'acc_grads': [], 'train_step_fn': None},
            'Y': {'vars': [Y, Bi], 'acc_grads': [], 'train_step_fn': None},
        }

        for p, stuff_p in phaseVariables.items():
            stuff_p['vars'] = list(filter(lambda x: x is not None, stuff_p['vars']))

        for p, stuff_p in phaseVariables.items():
            for q, stuff_q in phaseVariables.items():
                for v in stuff_q['vars']: v.trainable = False
            for v in stuff_p['vars']: v.trainable = True
            acc_grads = []
            for weight in model.trainable_weights:
                shape = weight.shape
                acc_grads.append(tf.Variable(shape=shape, initial_value=np.zeros(shape=shape, dtype='float32')))
            stuff_p['acc_grads'] = acc_grads
            stuff_p['train_step_fn'] = tf.function(experimental_relax_shapes=True)(train_step)  # for retrace

        for step, (x, y) in enumerate(epoch_train_data):

            epoch = step//steps_per_epoch
            batch = step % steps_per_epoch

            # start of epoch operations
            if batch == 0: progbar = generic_utils.Progbar(steps_per_epoch*batchsize)
            phase = ['X', 'Y'][epoch%2]

            for p, stuff in phaseVariables.items():
                for v in stuff['vars']: v.trainable = False
            stuff = phaseVariables[phase]
            for v in stuff['vars']: v.trainable = True
            acc_grads = stuff['acc_grads']
            train_step_fn = stuff['train_step_fn']  # for retrace

            u_in = x['u_in']
            i_in = x['i_in']
            rhat = y['rhat']

            cur_batchsize.assign(float(u_in.shape[0]))
            train_step_fn(model, u_in, i_in, rhat, loss_fn, acc_grads, acc_loss)
            num_seen.assign_add(cur_batchsize)
            progbar.add(batchsize, values=[(phase + ' loss', acc_loss)])

            # end of epoch operations
            if batch == steps_per_epoch-1:

                trace[epoch] += acc_loss
                num_seen.assign(0.)
                acc_loss.assign(0.)
                opt.apply_gradients(zip(acc_grads, model.trainable_weights))
                for acc_grad in acc_grads:
                    acc_grad.assign(tf.zeros(shape=acc_grad.shape, dtype=acc_grad.dtype))

                for callback in callbacks: callback.on_epoch_end(epoch)
                self.__save_config_cb(epoch)

        for callback in callbacks: callback.on_train_end()
        self.__env.get_state().update({'model': model})
        return trace

    def predict(self, u_in=None, i_in=None):
        self.__initialize()
        model = self.__env.get_state()['model']
        return model.predict({'u_in': u_in, 'i_in': i_in}, batch_size=5000)

class LMFEnv(Environment, LMFGD, SimpleMFRecAlgo, AUCEval):

    def __init__(
            self, path, model, data, state,
            epochs=30, batchsize=100, alpha=40,
            med_score=3.0, extra_callbacks=None
    ):
        Environment.__init__(self, path, model, data, state)
        if extra_callbacks is None: extra_callbacks = []
        extra_callbacks += [EvalCallback(self, "eval.csv", self)]
        LMFGD.__init__(self, self, epochs, batchsize=batchsize, alpha=alpha, callbacks=extra_callbacks)
        SimpleMFRecAlgo.__init__(self, self, self, output_key=0)
        AUCEval.__init__(self, self, self, med_score)

if __name__=="__main__":

    from reclibwh.utils.ItemMetadata import ExplicitDataFromCSV
    data_folder = '/home/ong/personal/recommender/data/ml-latest-small-2'
    d = ExplicitDataFromCSV(True, data_folder=data_folder)
    Utrain, Utest = d.make_training_datasets(dtype='sparse')

    # TODO: wrap in a factory function or do this inside of Algo
    # it = SparseMatRowIterator(2, padded=False, negative=True)({'S': Utrain, 'pad_val': -1.})
    # rename = Rename({'rows': 'u_in', 'cols': 'i_in', 'val': 'rhat'})(it)
    # train_mfit = XyDataIterator(ykey='rhat')(rename)

    train_mfit = LMF_data_iter_preset(Utrain)

    # TODO: wrap in a factory function or do this inside of Algo
    # auc_test_data = SparseMatRowIterator(10, padded=True, negative=False)({'S': Utrain, 'pad_val': -1., 'rows': np.arange(0,d.N,d.N//300)})
    # auc_train_data = SparseMatRowIterator(10, padded=True, negative=False)({'S': Utest, 'pad_val': -1., 'rows': np.arange(0,d.N,d.N//300)})

    auc_test_data = AUC_data_iter_preset(Utest, rows=np.arange(0,d.N,d.N//300))
    auc_train_data = AUC_data_iter_preset(Utrain, rows=np.arange(0,d.N,d.N//300))

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

    model_folder = '/home/ong/personal/recommender/models/test'
    save_path = os.path.join(model_folder, "LMF_{:s}".format(datetime.now().strftime("%Y-%m-%d.%H-%M-%S")))
    if not os.path.exists(save_path): os.mkdir(save_path)

    lmf = LMFEnv(save_path, m , data, env_vars, epochs=10)
    lmf.fit()
