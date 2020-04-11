import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import generic_utils
from scipy import sparse as sps
from tqdm import tqdm
import os
from keras.callbacks import Callback
from typing import Union, Optional


class ALS:

    def __init__(
            self, mode='train', N=100, M=100, K=10, lamb=1e-06, alpha=40., steps=10, model_path=None,
            Xinit=None, Yinit=None
    ):
        self.R = None #U # utility |user|x|items|, sparse, row major
        self.mode = mode
        self.model_path = model_path
        self.M = M
        self.N = N
        self.steps = steps
        if mode == 'train':
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
            self.lamb = lamb
            self.alpha = alpha
        else:
            assert model_path is not None, "model path required in predict mode"
            self.X = np.load(os.path.join(model_path, 'X.npy'))
            self.Y = np.load(os.path.join(model_path, 'Y.npy'))

    def _run_single_step(self, Y, X, C, R, p_float):

        assert self.mode == 'train', "cannot call when mode is not train"

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

        assert self.mode == 'train', "cannot call when mode is not train"

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

        assert self.mode == 'train', "cannot call when mode is not train"

        lamb = self.lamb
        p = _p.copy().toarray()
        R = _R.copy().toarray()
        C = _C.copy().toarray()
        loss = np.sum(np.multiply(C, np.square(p - np.matmul(X, Y.T))))
        loss += lamb*(np.mean(np.linalg.norm(X, 2, axis=1)) + np.mean(np.linalg.norm(Y, 2, axis=1)))
        return loss

    def train(self, U, cb:Callback=None):
        steps = self.steps
        assert self.mode == 'train', "cannot call when mode is not train"
        R = U
        p = R.astype(np.bool, copy=True).astype(np.float, copy=True)
        C = R.copy()
        C.data = 1 + self.alpha*C.data

        trace = np.zeros(shape=(steps,))

        for i in tqdm(range(steps)):

            Xp = self._run_single_step(self.Y, self.X, C, R, p)
            trace[i] = np.mean(np.abs(self.X - Xp))
            self.X = Xp
            Yp = self._run_single_step(self.X, self.Y, C.T, R.T, p.T)
            trace[i] += np.mean(np.abs(self.Y - Yp))
            self.Y = Yp

            self.save_as_epoch(i)

        return trace

    def save(self, model_path=None):

        if model_path is None:
            model_path = self.model_path
        assert model_path is not None, "model path not specified"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        np.save(os.path.join(model_path, "X.npy"), self.X)
        np.save(os.path.join(model_path, "Y.npy"), self.Y)

    def save_as_epoch(self, epoch:Union[str, int]='best'):
        model_name = 'epoch-{:03d}' if type(epoch)==int else 'epoch-{:s}'
        self.save(os.path.join(self.model_path, model_name.format(epoch)))


def get_and_pad_ratings(R, users, M, batchsize=None):
    ru = R[users, :]
    l = np.max(ru.getnnz(axis=1))
    if batchsize is None:
        batchsize = len(users)
    rp = np.zeros(shape=(batchsize, l), dtype=np.float)
    yp = M*np.ones(shape=(batchsize, l), dtype=np.int)
    for u, user in enumerate(users):
        rp[u, :ru[u].data.shape[0]] = ru[u].data
        yp[u, :ru[u].indices.shape[0]] = ru[u].indices
    return rp, yp


@tf.function(experimental_relax_shapes=True)
def als_run_regression(
        Y2: tf.Tensor, Lamb: tf.Tensor, Yp: tf.Tensor, r: tf.Tensor, alpha: tf.constant,
        batchsize: int, k: int):

    Ypp = tf.reshape(Yp, [batchsize, -1, k])
    rp = tf.reshape(r, [batchsize, -1, 1])

    c = np.float64(1.) + tf.multiply(alpha, rp)
    p = tf.cast(tf.math.greater(rp, 0.), tf.float64)

    L = Y2 + tf.linalg.matmul(Ypp, alpha * tf.multiply(rp, Ypp), transpose_a=True)
    Linv = tf.linalg.inv(L + Lamb)
    x = tf.linalg.matmul(Linv, tf.linalg.matmul(Ypp, tf.multiply(c, p), transpose_a=True))
    X = tf.reshape(x, [batchsize, k])

    return X


class ALSTF(ALS):

    def __init__(self, batchsize, *args, **kwargs):
        self.batchsize = batchsize
        super(ALSTF, self).__init__(*args, **kwargs)

        # variables for online training
        self.Y2Tensor = None
        self.LambTensor = None
        self.YTensor = None
        self.XTensor = None

    def _add_users(self, num=1):
        self.X = np.append(self.X, np.zeros(shape=(num,self.K)), axis=0)
        if self.XTensor is not None: # update cached tensor if exists
            self.XTensor = tf.Variable(
                name="X", dtype=tf.float64,
                initial_value=np.append(self.X, np.zeros(shape=(1, self.K)), axis=0)
            )
        self.N += num

    def train_update(self, U, users, cb:Callback=None, use_cache=False):

        if cb: cb.on_epoch_end(0) # eval before

        users = np.unique(users)  # remove duplicates
        assert np.max(users) < self.N, "new user exceeds embedding dimensions {}, X size: {:d}".\
            format(users, self.N)

        if self.YTensor is None or not use_cache:
            Y = tf.Variable(name="Y", dtype=tf.float64,
                            initial_value=np.append(self.Y, np.zeros(shape=(1, self.K)), axis=0))
            if use_cache: self.YTensor = Y
        else: Y = self.YTensor
        if self.XTensor is None or not use_cache:
            X = tf.Variable(name="X", dtype=tf.float64,
                            initial_value=np.append(self.X, np.zeros(shape=(1, self.K)), axis=0))
            if use_cache: self.XTensor = X
        else: X = self.XTensor
        trace = self._run_single_step(Y, X, U, users, prefix="X update", use_cache=True)
        self.X = X.numpy()[:-1]
        self.Y = Y.numpy()[:-1]

        if cb: cb.on_epoch_end(1)  # eval after
        return [0, trace]

    def _run_single_step(self, Y: tf.Tensor, X: tf.Tensor, R: sps.csr_matrix, users:Optional[np.array]=None, prefix="", use_cache=False):

        batchsize = self.batchsize
        N = X.shape[0]-1
        M = Y.shape[0]-1
        k = self.K
        lamb = self.lamb
        alpha = self.alpha

        assert self.mode == 'train', "cannot call when mode is not train"

        if users is None: users = np.arange(0,N)
        if self.Y2Tensor is None or not use_cache:
            Y2 = tf.reshape(tf.tensordot(tf.transpose(Y), Y, axes=1, name='Y2'), shape=[1, k, k])
            if use_cache: self.Y2Tensor = Y2
        else: Y2 = self.Y2Tensor
        if self.LambTensor is None or not use_cache:
            Lamb = tf.multiply(tf.constant(lamb, dtype=tf.float64), tf.linalg.eye(num_rows=k, batch_shape=[batchsize], dtype=tf.float64))
            if use_cache: self.LambTensor = Lamb
        else: Lamb = self.LambTensor

        diff = tf.Variable(dtype=tf.float64, initial_value=0.)
        batch_diff = tf.Variable(dtype=tf.float64, initial_value=0.)

        progbar = generic_utils.Progbar(np.ceil(float(N)/batchsize))

        for i in range(0, len(users), batchsize):

            us = users[i:min(i + batchsize, N)] #np.arange(i, )
            rs, ys = get_and_pad_ratings(R, us, M, batchsize=batchsize)

            Yp = tf.nn.embedding_lookup(Y, ys)

            Xnew = als_run_regression(
                Y2, Lamb, Yp,
                tf.constant(rs, tf.float64), tf.constant(alpha, tf.float64),
                tf.constant(batchsize), tf.constant(k)
            )

            us = tf.convert_to_tensor(us, dtype=tf.int32) # need to convert to tensor to index tensors
            batch_diff.assign(tf.reduce_sum(tf.abs(tf.gather(X, us)-Xnew[:len(us)])))
            us = tf.expand_dims(us, axis=1)
            X.scatter_nd_update(us, Xnew[:len(us)])

            diff.assign_add(batch_diff)
            progbar.add(1, values=[(prefix + ' embedding delta', batch_diff/min(i + batchsize, N))])

        return diff / tf.cast(N, dtype=tf.float64)

    def train(self, U, cb:Callback=None):

        assert self.mode == 'train', "cannot call when mode is not train"

        k = self.K
        M = self.M
        N = self.N
        steps = self.steps

        # padding for reasons
        X = tf.Variable(name="X", dtype=tf.float64, initial_value=np.append(self.X, np.zeros(shape=(1,self.K)), axis=0))
        Y = tf.Variable(name="Y", dtype=tf.float64, initial_value=np.append(self.Y, np.zeros(shape=(1,self.K)), axis=0))

        Y[M].assign(tf.zeros([k], dtype=tf.float64))
        X[N].assign(tf.zeros([k], dtype=tf.float64))

        UT = sps.csr_matrix(U.T)
        trace = np.zeros(shape=(steps,))

        for i in range(steps):
            print("step {}/{}".format(i, steps-1))
            dX = self._run_single_step(Y, X, U, prefix="X")
            dY = self._run_single_step(X, Y, UT, prefix="Y")

            trace[i] = .5*(dX + dY)

            self.X = X.numpy()[:-1] # remove padded dimension
            self.Y = Y.numpy()[:-1]
            self.save_as_epoch(i)
            if cb:
                cb.on_epoch_end(i)

        self.X = X.numpy()[:-1]
        self.Y = Y.numpy()[:-1]

        pd.DataFrame({
            'epoch': list(range(steps)), 'trace': trace,
        }).to_csv(os.path.join(self.model_path, 'trace.csv'), index=False)

        return trace

    def save(self, model_path=None):

        if model_path is None:
            model_path = self.model_path
        assert model_path is not None, "model path not specified"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        np.save(os.path.join(model_path, "X.npy"), self.X[:-1])
        np.save(os.path.join(model_path, "Y.npy"), self.Y[:-1])

    def save_as_epoch(self, epoch:Union[str, int]='best'):
        model_name = 'epoch-{:03d}' if type(epoch)==int else 'epoch-{:s}'
        self.save(os.path.join(self.model_path, model_name.format(epoch)))


@tf.function
def run_single_step(
    batchsize: int, N: int, M: int, k: int, alpha: int, lamb: float,
    Y: tf.Tensor, X: tf.Tensor, diff: tf.Tensor,
    data: tf.data.Dataset
):
    """
    tf function version of the _run_single_step method in ALSTF, using tensorflow datasets
    to iterate through the sparse matrix
    :param batchsize:
    :param N:
    :param M:
    :param k:
    :param alpha:
    :param lamb:
    :param Y:
    :param X:
    :param diff:
    :param data:
    :return:
    """

    batchsize_c = tf.constant(batchsize)
    N_c = tf.constant(N)

    Y2 = tf.reshape(tf.tensordot(tf.transpose(Y), Y, axes=1, name='Y2'), shape=[1, K, K])
    Lamb = tf.multiply(tf.constant(lamb, dtype=tf.float64),
                       tf.linalg.eye(num_rows=K, batch_shape=[batchsize], dtype=tf.float64))

    for i, (rs, ys) in tf.data.Dataset.zip(
        (tf.data.Dataset.from_tensor_slices(tf.range(0, N, batchsize, dtype=tf.int32)),
         data)
    ):
        Yp = tf.nn.embedding_lookup(Y, ys)

        Xnew = als_run_regression(
            Y2, Lamb, Yp,
            rs, tf.constant(alpha, tf.float64),
            tf.constant(batchsize), tf.constant(k)
        )
        tf.print(ys.shape)
        if N_c < i + batchsize_c:
            diff.assign_add(tf.reduce_sum(tf.abs(X[i:N_c] - Xnew[:N_c - i])))
            X[i:N_c].assign(Xnew[:N_c - i])
        else:
            diff.assign_add(tf.reduce_sum(tf.abs(X[i:i + batchsize_c] - Xnew)))
            X[i:i + batchsize_c].assign(Xnew)

    return diff / tf.cast(N_c, dtype=tf.float64)


####### Pure tf.function implementation with tf.datasets #######
# empirically tested to be slower with ml-20m with in-memory sparse matrices

def sparse2dataset(indices: np.array, data: np.array, N: int, M: int, batch_size: int):

    ds = tf.data.Dataset.from_tensor_slices(
        tf.sparse.reorder(
            tf.sparse.SparseTensor(indices=indices, values=data, dense_shape=(N, M))
        )
    )

    def iter_r():
        for d in ds:
            yield d.values
        return

    def iter_y():
        for d in ds:
            yield d.indices[:,0]
        return

    rs = tf.data.Dataset.from_generator(iter_r, output_types=(tf.float64))
    ys = tf.data.Dataset.from_generator(iter_y, output_types=(tf.int64))
    rs = rs.padded_batch(batch_size=batch_size, padded_shapes=[None], padding_values=np.float64(0), drop_remainder=True)
    ys = ys.padded_batch(batch_size=batch_size, padded_shapes=[None], padding_values=np.int64(M), drop_remainder=True)

    return rs, ys


class ALSTF_DS(ALSTF):

    """
    ALS using tf dataset api, just to see if faster
    """

    def __init__(self, *args, **kwargs):
        super(ALSTF_DS, self).__init__(*args, **kwargs)

    def train(self, U, cb:Callback=None):

        rows, cols, data = sps.find(U)
        indices = np.zeros(shape=(len(c), 2), dtype=np.int64)
        indices[:, 0] = rows
        indices[:, 1] = cols

        assert self.mode == 'train', "cannot call when mode is not train"

        k = self.K
        M = self.M
        N = self.N
        steps = self.steps
        batchsize = self.batchsize
        lamb = self.lamb
        alpha = self.alpha

        X = tf.Variable(name="X", dtype=tf.float64, initial_value=self.X)
        Y = tf.Variable(name="Y", dtype=tf.float64, initial_value=self.Y)
        diff = tf.Variable(dtype=tf.float64, initial_value=0.)

        Y[M].assign(tf.zeros([k], dtype=tf.float64))
        X[N].assign(tf.zeros([k], dtype=tf.float64))

        trace = np.zeros(shape=(steps,))

        progbar = generic_utils.Progbar(np.ceil(float(N) / batchsize))

        for i in range(steps):
            diff.assign(0.)

            rs, ys = sparse2dataset(indices, data, N, M, batchsize)
            ds = tf.data.Dataset.zip((rs, ys))
            dX = run_single_step(batchsize, N, M, K, alpha, lamb, Y, X, diff, ds)

            rs, ys = sparse2dataset(indices[:, ::-1], data, M, N, batchsize)
            ds = tf.data.Dataset.zip((rs, ys))
            dY = run_single_step(batchsize, M, N, K, alpha, lamb, X, Y, diff, ds)

            trace[i] += .5*(dX + dY)
            progbar.add(1, [("embedding deltas", trace[i])])

            self.X = X.numpy()
            self.Y = Y.numpy()
            self.save(os.path.join(self.model_path, 'epoch-{:03d}'.format(i)))

            if cb:
                cb.on_epoch_end(i)

        pd.DataFrame({
            'epoch': list(range(steps)), 'trace': trace,
        }).to_csv(os.path.join(self.model_path, 'trace.csv'), index=False)

        return trace



if __name__=="__main__":

    M = 100
    N = 50
    P = 150
    np.random.seed(42)
    data = np.random.randint(1, 10, size=(P,))
    c = np.random.randint(0, M*N, size=(P,))
    rows = c//M
    cols = c%M

    U = sps.csr_matrix((data, (rows, cols)), shape=(N,M))
    R = U

    K = 5

    Xinit = np.random.normal(0, 1 / np.sqrt(K), size=(N, K))
    Yinit = np.random.normal(0, 1 / np.sqrt(K), size=(M, K))

    p = R.copy().astype(np.bool).astype(np.float, copy=True)
    C = R.copy()
    C.data = 1 + 1. * C.data
    als = ALS(N=N, M=M, K=K)
    als._run_single_step(Yinit, Xinit, C, R, p)