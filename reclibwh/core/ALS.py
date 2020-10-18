import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import generic_utils
from scipy import sparse as sps
from tqdm import tqdm
import os
from typing import Union, Optional
import json
from tensorflow.keras.models import Model
from .Models import STANDARD_KERAS_SAVE_FMT, initialize_from_json, model_restore
from datetime import datetime
from ..data.iterators import EpochIterator, Rename, XyDataIterator, SparseMatRowIterator, Normalizer
from .EvalProto import EvalCallback, AUCEval
from .RecAlgos import SimpleMFRecAlgo, RecAlgo
from .Environment import Environment, Algorithm, UpdateAlgo
from reclibwh.data.PresetIterators import ALS_data_iter_preset, AUC_data_iter_preset

class ALS:

    """
    reference implementation for debugging pusposes
    """

    def __init__(
            self, mode='train', N=100, M=100, K=10, lamb=1e-06, alpha=40., steps=10,
            Xinit=None, Yinit=None
    ):

        self.R = None #U # utility |user|x|items|, sparse, row major
        self.mode = mode
        self.M = M
        self.N = N
        self.steps = steps
        self.lamb = lamb
        self.alpha = alpha
        self.K = K

        # dense feature matrices
        if Xinit is not None:
            self.X = Xinit.copy()
        else:
            np.random.seed(42)
            self.X = np.random.normal(0, 1 / np.sqrt(self.K), size=(N, self.K))
        if Yinit is not None:
            self.Y = Yinit.copy()
        else:
            np.random.seed(42)
            self.Y = np.random.normal(0, 1/np.sqrt(self.K), size=(M, self.K))

    def _run_single_step(self, Y, X, C, R, p_float):

        lamb = self.lamb
        Y2 = np.tensordot(Y.T, Y, axes=1)
        Lamb = lamb*np.eye(self.K)
        alpha = self.alpha
        Xp = X.copy()

        for u, x in enumerate(X):
            cu = C[u, :]
            ru = R[u, :]
            p_idx = sps.find(ru)[1]
            Yp = Y[p_idx, :]
            rp = ru[:, p_idx].toarray()[0]
            L = np.linalg.inv(Y2 + np.matmul(Yp.T, alpha*np.multiply(np.expand_dims(rp, axis=1), Yp)) + Lamb)
            cup = cu[:, p_idx].toarray()[0]
            p = p_float[u, :]
            pp = p[:, p_idx].toarray()[0]
            xp = np.matmul(L, np.tensordot(Yp.T, np.multiply(cup, pp), axes=1))
            Xp[u, :] = xp
        return Xp

    def _run_single_step_naive(self, Y, X, _R, p_float):

        lamb = self.lamb
        Lamb = lamb * np.eye(self.K)
        alpha = self.alpha

        C = _R.copy().toarray()
        C = 1 + alpha*C
        P = p_float.copy().toarray()

        Xp = X.copy()

        for u, x in enumerate(X):
            cu = C[u, :]
            Cu = np.diag(cu)
            pu = P[u,:]
            L = np.linalg.inv(np.matmul(Y.T, np.matmul(Cu, Y)) + Lamb)
            xp = np.matmul(L, np.matmul(Y.T,np.matmul(Cu, pu)))
            Xp[u, :] = xp

        return Xp

    def _calc_loss(self, Y, X, _C, _R, _p):

        lamb = self.lamb
        p = _p.copy().toarray()
        R = _R.copy().toarray()
        C = _C.copy().toarray()
        loss = np.sum(np.multiply(C, np.square(p - np.matmul(X, Y.T))))
        loss += lamb*(np.mean(np.linalg.norm(X, 2, axis=1)) + np.mean(np.linalg.norm(Y, 2, axis=1)))
        return loss

    def train(self, U):
        steps = self.steps
        assert self.mode == 'train', "cannot call when mode is not train"
        R = U
        p = R.astype(np.bool, copy=True).astype(np.float, copy=True)
        C = R.copy()
        C.data = 1 + self.alpha*C.data

        start = -1
        trace = np.zeros(shape=(steps,))

        for i in tqdm(range(start+1,steps)):

            Xp = self._run_single_step(self.Y, self.X, C, R, p)
            trace[i] = np.mean(np.abs(self.X - Xp))
            self.X = Xp
            Yp = self._run_single_step(self.X, self.Y, C.T, R.T, p.T)
            trace[i] += np.mean(np.abs(self.Y - Yp))
            self.Y = Yp
        return trace

    def save(self, model_path=None):

        assert model_path is not None, "model path not specified"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        np.save(os.path.join(model_path, "X.npy"), self.X)
        np.save(os.path.join(model_path, "Y.npy"), self.Y)

    def load_model(self, model_path):
        self.X = np.load(os.path.join(model_path, 'X.npy'))
        self.Y = np.load(os.path.join(model_path, 'Y.npy'))
        self.K = self.X.shape[1]

    def load_trace(self, trace_path):
        return np.array(pd.read_csv(trace_path)['trace'])


@tf.function(experimental_relax_shapes=True)
def als_run_regression(
        Y2: tf.Tensor, Lamb: tf.Tensor, Yp: tf.Tensor, r: tf.Tensor, alpha: tf.constant,
        batchsize: tf.constant, k: tf.constant):

    Ypp = tf.reshape(Yp, [batchsize, -1, k])
    rp = tf.reshape(r, [batchsize, -1, 1])

    c = np.float32(1.) + tf.multiply(alpha, rp)
    p = tf.cast(tf.math.greater(rp, 0.), tf.float32)

    L = Y2 + tf.linalg.matmul(Ypp, alpha * tf.multiply(rp, Ypp), transpose_a=True)
    Linv = tf.linalg.inv(L + Lamb)
    x = tf.linalg.matmul(Linv, tf.linalg.matmul(Ypp, tf.multiply(c, p), transpose_a=True))
    X = tf.reshape(x, [batchsize, k])

    return X

@tf.function(experimental_relax_shapes=True)
def run_als_step(
        X: tf.Tensor, Y: tf.Tensor, Y2: tf.Tensor, LambdaI: tf.Tensor, # parameters
        us: tf.Tensor, ys: tf.Tensor, rs: tf.Tensor, # data
        alpha: float
) -> tf.Tensor:

    Yp = tf.nn.embedding_lookup(Y, ys)
    k = tf.shape(Y)[1]
    batchsize = tf.shape(us)[0]

    Xnew = als_run_regression(
        Y2, LambdaI, Yp,
        rs, tf.constant(alpha, tf.float32),
        batchsize, k
    )

    diff = tf.reduce_sum(tf.abs(tf.gather(X, us) - Xnew[:batchsize]))
    us = tf.expand_dims(us, axis=1)
    X.scatter_nd_update(us, Xnew[:batchsize])

    return diff

def run_als_epoch(
        als: Model, alsc: Model,
        Xname: str, Yname: str, Y2name: str,
        data, alpha: float, prefix="",
) -> tf.Tensor:

    X = als.get_layer(Xname).trainable_weights[0]
    Y = als.get_layer(Yname).trainable_weights[0]
    Y2 = alsc.get_layer(Y2name).trainable_weights[0]
    LambdaI = alsc.get_layer("LambdaI").trainable_weights[0]

    progbar = generic_utils.Progbar(len(data))
    N = X.shape[0]
    diff = tf.Variable(dtype=tf.float32, initial_value=0.)
    n = tf.Variable(dtype=tf.int32, initial_value=0)

    for x, y in data:

        # todo: put this in an iterator maybe
        us = tf.convert_to_tensor(x['u_in'], dtype=tf.int32)
        ys = tf.convert_to_tensor(x['i_in'], dtype=tf.int32)
        rs = tf.convert_to_tensor(y['rhat'], dtype=tf.float32)

        batch_size = us.shape[0]
        n.assign_add(batch_size)

        batch_diff = run_als_step(X, Y, Y2, LambdaI, us, ys, rs, alpha)
        diff.assign_add(batch_diff)
        progbar.add(1, values=[(prefix + ' embedding delta', diff/tf.cast(n*N, tf.float32))])

        # update shenanigans

    return diff/tf.cast(n*N, tf.float32)

def als_update_cache(als: Model, alsc: Model, Yname, Y2name, lamb=1e-06):

    Y = als.get_layer(Yname).trainable_weights[0]
    Y2 = alsc.get_layer(Y2name).trainable_weights[0]
    LambdaI = alsc.get_layer("LambdaI").trainable_weights[0]

    k = LambdaI.shape[0]

    Y2.assign(tf.reshape(tf.tensordot(tf.transpose(Y), Y, axes=1), shape=[k, k]))
    LambdaI.assign(lamb*tf.eye(k))

    return

class ALSTrainer(Algorithm):

    def __init__(self, env: Environment, epochs=30, alpha=40.0, lamb=1e-06, extra_callbacks=None):
        config = {}
        config['epochs'] = epochs
        config['start_epoch'] = 0
        config['alpha'] = alpha
        config['lamb'] = lamb
        self.__config = config
        self.__extra_callbacks = extra_callbacks
        self.__initialized = False
        self.__als_model = None
        self.__env = env

    def __env_save(self):
        state = self.__env.get_state()
        path = state['environment_path']
        save_fmt = state.get('save_fmt', STANDARD_KERAS_SAVE_FMT)
        val_loss = state.get('val_loss', 0)
        epoch = self.__config['start_epoch']
        self.__als_model.save(os.path.join(path, save_fmt.format(epoch=epoch, val_loss=val_loss)))
        with open(os.path.join(path, 'algorithm_config.json'), 'w') as fp: json.dump(self.__config, fp)

    def __save_config_cb(self, epoch):
        self.__config['start_epoch'] = epoch
        self.__env_save()

    def __env_restore(self):
        state = self.__env.get_state()
        path = state['environment_path']
        if not os.path.exists(os.path.join(path, 'algorithm_config.json')):
            print("algorithm config does not exist. cannot restore")
        else:
            with open(os.path.join(path, 'algorithm_config.json'), 'r') as fp:
                self.__config = json.load(fp)
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
        models = state['model']
        als = models[0]
        alsc = models[1]
        data = state['data']
        self.__als_model = als
        # optional

        alpha = self.__config['alpha']
        lamb = self.__config['lamb']
        start_epoch = self.__config['start_epoch']
        epochs = self.__config['epochs']
        callbacks = self.__extra_callbacks

        train_data_X = data['train_data_X']
        train_data_Y = data['train_data_Y']

        for epoch in range(start_epoch, epochs):

            als_update_cache(als, alsc, 'Y', 'Y2', lamb)
            run_als_epoch(als, alsc, 'X', 'Y', 'Y2', train_data_X, alpha, prefix="X")

            als_update_cache(als, alsc, 'X', 'X2', lamb)
            run_als_epoch(als, alsc, 'Y', 'X', 'X2', train_data_Y, alpha, prefix="Y")

            # self.env.set_state({'val_loss': float(epoch)})
            for callback in callbacks: callback.on_epoch_end(epoch)
            self.__save_config_cb(epoch)

        for callback in callbacks: callback.on_train_end()

        return

    def predict(self, u_in=None, i_in=None):
        self.__initialize()
        model = self.__env.get_state()['model'][0]
        return model.predict({'u_in': u_in, 'i_in': i_in}, batch_size=5000)

    def __evaluate(self, data=None):
        self.__initialize()
        model = self.__env.get_state()['model']
        return model.evaluate(EpochIterator(1)(data).__iter__(), batch_size=5000)

class ALSUpdateAlgo(UpdateAlgo):

    def __init__(self, env: Environment, algo: ALSTrainer):
        self.__algo = algo
        self.__env = env

    def update_user(self, data):

        state = self.__env.get_state()
        # mandatory
        algo_config = self.__algo._ALSTrainer__config
        models = state['model']
        als = models[0]
        alsc = models[1]
        alpha = algo_config['alpha']

        run_als_epoch(als, alsc, 'X', 'Y', 'Y2', data, alpha, prefix="X")

        pass

class ALSEnv(Environment, ALSTrainer, ALSUpdateAlgo, SimpleMFRecAlgo, AUCEval):

    def __init__(
            self, path, model, data, state,
            epochs=30, alpha=40.0, lamb=1e-06, extra_callbacks=None,
            med_score=3.0
    ):

        Environment.__init__(self, path, model, data, state)
        if extra_callbacks is None: extra_callbacks = []
        extra_callbacks += [EvalCallback(self, "eval.csv", self)]
        ALSTrainer.__init__(self, self, epochs=epochs, alpha=alpha, lamb=lamb, extra_callbacks=extra_callbacks)
        SimpleMFRecAlgo.__init__(self, self, self, output_key=0)
        AUCEval.__init__(self, self, self, med_score)
        ALSUpdateAlgo.__init__(self, self, self)

if __name__=="__main__":

    # M = 800
    # N = 350
    # K = 20
    # P = 1500
    # train_val_split = 0.8
    #
    # alpha = 40.
    # lamb = 1e-06
    # np.random.seed(42)
    # data = np.random.randint(1, 10, size=(P,))
    # c = np.random.randint(0, M * N, size=(P,))
    # c = np.unique(c)
    # data = data[:len(c)]
    # rows = c // M
    # cols = c % M
    #
    # data_train = data[:int(train_val_split * len(data))]
    # data_test = data[int(train_val_split * len(data)):]
    # rows_train = rows[:int(train_val_split * len(data))]
    # rows_test = rows[int(train_val_split * len(data)):]
    # cols_train = cols[:int(train_val_split * len(data))]
    # cols_test = cols[int(train_val_split * len(data)):]
    #
    # Utrain = sps.csr_matrix((data_train, (rows_train, cols_train)), shape=(N, M), dtype=np.float64)
    # Utest = sps.csr_matrix((data_test, (rows_test, cols_test)), shape=(N, M), dtype=np.float64)
    #
    # Xinit = np.random.normal(0, 1 / np.sqrt(K), size=(N, K))
    # Yinit = np.random.normal(0, 1 / np.sqrt(K), size=(M, K))
    # Yinit_a = np.append(Yinit, np.zeros(shape=(1, K)), axis=0)
    # Xinit_a = np.append(Xinit, np.zeros(shape=(1, K)), axis=0)

    # R = U
    #
    # K = 5
    #
    # Xinit = np.random.normal(0, 1 / np.sqrt(K), size=(N, K))
    # Yinit = np.random.normal(0, 1 / np.sqrt(K), size=(M, K))
    #
    # p = R.copy().astype(np.bool).astype(np.float, copy=True)
    # C = R.copy()
    # C.data = 1 + 1. * C.data
    # als = ALS(N=N, M=M, K=K)
    # als._run_single_step(Yinit, Xinit, C, R, p)

    from reclibwh.utils.ItemMetadata import ExplicitDataFromCSV

    data_folder = '/home/ong/personal/recommender/data/ml-latest-small-2'
    d = ExplicitDataFromCSV(True, data_folder=data_folder)
    Utrain, Utest = d.make_training_datasets(dtype='sparse')

    M = d.M + 1
    N = d.N + 1

    model_folder = '/home/ong/personal/recommender/models/test'
    save_path = os.path.join(model_folder, "ALS_{:s}".format(datetime.now().strftime("%Y-%m-%d.%H-%M-%S")))
    if not os.path.exists(save_path): os.mkdir(save_path)

    # train_data_X = SparseMatRowIterator(20, padded=True, negative=False)({'S': Utrain, 'pad_val': M})
    # train_data_X = Rename({'rows': 'u_in', 'cols': 'i_in', 'val': 'rhat'})(train_data_X)
    # train_data_X = XyDataIterator(ykey='rhat')(train_data_X)
    # train_data_Y = SparseMatRowIterator(20, padded=True, negative=False)({'S': sps.csr_matrix(Utrain.T), 'pad_val': N})
    # train_data_Y = Rename({'rows': 'u_in', 'cols': 'i_in', 'val': 'rhat'})(train_data_Y)
    # train_data_Y = XyDataIterator(ykey='rhat')(train_data_Y)

    train_data_X = ALS_data_iter_preset(Utrain, batchsize=20)
    train_data_Y = ALS_data_iter_preset(sps.csr_matrix(Utrain.T), batchsize=20)

    # auc_test_data = SparseMatRowIterator(10, padded=True, negative=False)({'S': Utest, 'pad_val': -1.})
    # auc_train_data = SparseMatRowIterator(10, padded=True, negative=False)({'S': Utrain, 'pad_val': -1.})

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

    m = initialize_from_json(data_conf={"M": M+1, "N": N+1}, config_path="ALS.json.template")
    # m[0].get_layer("X").set_weights([Xinit_a])
    # m[0].get_layer("Y").set_weights([Yinit_a])
    # alst = ALSTrainer(epochs=1)
    # rec = SimpleMFRecAlgo(output_key=0)
    # env = Environment(save_path, m, data, alst, state=env_vars, rec=rec)

    # auc_eval = AUCEval(3.0)
    # training_callbacks = [] #[EvalCallback(auc_eval, "auc.csv", env)]
    # env.set_state({'training_callbacks': training_callbacks})
    # env.run_train()

    # update algorithm test case
    # update_algo = ALSUpdateAlgo()
    # env.environment_add_object(update_algo, 'update_algo')

    alsenv = ALSEnv(save_path, m, data, env_vars, epochs=3)
    alsenv.fit()

    # update first and last user
    rows, cols, vals = sps.find(Utrain[0])
    rows = np.concatenate([rows, np.ones(dtype=int, shape=rows.shape)*(N-1)])
    cols = np.concatenate([cols, cols])
    vals = np.concatenate([vals, vals])
    Uupdate = sps.csr_matrix((vals, (rows, cols)), shape=(N,M))

    print(Uupdate)

    # update_data = SparseMatRowIterator(20, padded=True, negative=False)({'S': Uupdate, 'pad_val': M, 'rows': [0, N-1]})
    # update_data = Rename({'rows': 'u_in', 'cols': 'i_in', 'val': 'rhat'})(update_data)
    # update_data = XyDataIterator(ykey='rhat')(update_data)
    update_data = ALS_data_iter_preset(Uupdate, rows=[0, N-1])

    print(alsenv.get_state()['model'][0].get_layer('X').get_weights()[0][N-1])

    alsenv.update_user(update_data)

    print(alsenv.get_state()['model'][0].get_layer('X').get_weights()[0][0])
    print(alsenv.get_state()['model'][0].get_layer('X').get_weights()[0][N-1])
