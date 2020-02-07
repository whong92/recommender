import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import tensorflow as tf
from .ALS import ALS
from keras.utils import generic_utils
import pandas as pd

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
        if self.mode == 'train':
            self.X = np.append(self.X, np.zeros(shape=(1,self.K)), axis=0)
            self.Y = np.append(self.Y, np.zeros(shape=(1,self.K)), axis=0)

    def _run_single_step(self, Y: tf.Tensor, X: tf.Tensor, R: sps.csr_matrix, prefix=""):

        batchsize = self.batchsize
        N = X.shape[0]-1
        M = Y.shape[0]-1
        k = self.K
        lamb = self.lamb
        alpha = self.alpha

        assert self.mode == 'train', "cannot call when mode is not train"

        Y2 = tf.reshape(tf.tensordot(tf.transpose(Y), Y, axes=1, name='Y2'), shape=[1, k, k])
        Lamb = tf.multiply(tf.constant(lamb, dtype=tf.float64),
                           tf.linalg.eye(num_rows=k, batch_shape=[batchsize], dtype=tf.float64))
        diff = tf.Variable(dtype=tf.float64, initial_value=0.)
        batch_diff = tf.Variable(dtype=tf.float64, initial_value=0.)

        progbar = generic_utils.Progbar(np.ceil(float(N)/batchsize))

        for i in range(0, N, batchsize):

            us = np.arange(i, min(i + batchsize, N))
            rs, ys = get_and_pad_ratings(R, us, M, batchsize=batchsize)

            Yp = tf.nn.embedding_lookup(Y, ys)

            Xnew = als_run_regression(
                Y2, Lamb, Yp,
                tf.constant(rs, tf.float64), tf.constant(alpha, tf.float64),
                tf.constant(batchsize), tf.constant(k)
            )

            if N < i + batchsize:
                batch_diff.assign(tf.reduce_sum(tf.abs(X[i:N]-Xnew[:N - i])))
                X[i:N].assign(Xnew[:N - i])
            else:
                batch_diff.assign(tf.reduce_sum(tf.abs(X[i:i + batchsize]-Xnew)))
                X[i:i + batchsize].assign(Xnew)

            diff.assign_add(batch_diff)
            progbar.add(1, values=[(prefix + ' embedding delta', batch_diff/min(i + batchsize, N))])

        return diff / tf.cast(N, dtype=tf.float64)

    def train(self, U, steps=10):

        assert self.mode == 'train', "cannot call when mode is not train"

        k = self.K
        M = self.M
        N = self.N

        X = tf.Variable(name="X", dtype=tf.float64, initial_value=self.X)
        Y = tf.Variable(name="Y", dtype=tf.float64, initial_value=self.Y)

        Y[M].assign(tf.zeros([k], dtype=tf.float64))
        X[N].assign(tf.zeros([k], dtype=tf.float64))

        UT = sps.csr_matrix(U.T)

        trace = np.zeros(shape=(steps,))

        for i in range(steps):
            print("step {}/{}".format(i, steps-1))
            dX = self._run_single_step(Y, X, U, prefix="X")
            dY = self._run_single_step(X, Y, UT, prefix="Y")

            trace[i] = .5*(dX + dY)

            self.X = X.numpy()
            self.Y = Y.numpy()
            self.save(os.path.join(self.model_path, 'epoch-{:03d}'.format(i)))

        self.X = X.numpy()
        self.Y = Y.numpy()

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

####### Pure tf.function implementation with tf.datasets #######
# empirically tested to be slower with ml-20m with in-memory sparse matrices

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

    def train(self, U, steps=10):

        rows, cols, data = sps.find(U)
        indices = np.zeros(shape=(len(c), 2), dtype=np.int64)
        indices[:, 0] = rows
        indices[:, 1] = cols

        assert self.mode == 'train', "cannot call when mode is not train"

        k = self.K
        M = self.M
        N = self.N
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

        pd.DataFrame({
            'epoch': list(range(steps)), 'trace': trace,
        }).to_csv(os.path.join(self.model_path, 'trace.csv'), index=False)

        return trace


if __name__=="__main__":

    #########################
    # setup
    M = 800
    N = 350
    P = 1500
    alpha = 40.
    lamb = 1e-06
    np.random.seed(42)
    data = np.random.randint(1, 10, size=(P,))
    c = np.random.randint(0, M * N, size=(P,))
    c = np.unique(c)
    data = data[:len(c)]
    rows = c // M
    cols = c % M

    U = sps.csr_matrix((data, (rows, cols)), shape=(N,M), dtype=np.float64)
    R = U

    indices = np.zeros(shape=(len(c),2), dtype=np.int64)
    indices[:,0] = rows
    indices[:,1] = cols

    rs, ys = sparse2dataset(indices, data, N, M, 3)

    K = 5

    Xinit = np.random.normal(0, 1 / np.sqrt(K), size=(N, K))
    Yinit = np.random.normal(0, 1 / np.sqrt(K), size=(M, K))

    #########################
    # reference non-tf implementation
    p = R.copy().astype(np.bool).astype(np.float, copy=True)
    C = R.copy()
    C.data = 1 + alpha * C.data

    als = ALS(N=N, M=M, K=K, alpha=alpha, lamb=lamb, Xinit=Xinit, Yinit=Yinit)

    Xnew = als._run_single_step(Yinit, Xinit, C, R, p)
    print(Xnew[:3])
    Ynew = als._run_single_step(Xinit, Yinit, C.T, R.T, p.T)
    print(Ynew[:3])

    trace = als.train(R, steps=10)
    plt.plot(trace)
    plt.show()

    #########################
    # test TF implementation
    batchsize=3
    Yinit_a = np.append(Yinit, np.zeros(shape=(1,K)), axis=0)
    Xinit_a = np.append(Xinit, np.zeros(shape=(1, K)), axis=0)

    alstf = ALSTF(batchsize=300, N=N, M=M, K=K, alpha=alpha, lamb=lamb, Xinit=Xinit, Yinit=Yinit)

    X = tf.Variable(name="X", dtype=tf.float64, initial_value=Xinit_a)
    Y = tf.Variable(name="Y", dtype=tf.float64, initial_value=Yinit_a)

    alstf._run_single_step(Y, X, R)
    print(X[:3])
    print(Y[:3])

    trace = alstf.train(U)
    plt.plot(trace)
    plt.show()

    #########################
    # test tf implementaion with datasets
    Y = tf.Variable(initial_value=Yinit)
    X = tf.Variable(initial_value=Xinit)
    Y2 = tf.reshape(tf.tensordot(tf.transpose(Y), Y, axes=1, name='Y2'), shape=[1, K, K])
    Lamb = tf.multiply(tf.constant(lamb, dtype=tf.float64),
                       tf.linalg.eye(num_rows=K, batch_shape=[batchsize], dtype=tf.float64))
    ds = tf.data.Dataset.zip((rs, ys))
    diff = tf.Variable(dtype=tf.float64, initial_value=0.)
    trace = run_single_step(batchsize, N, M, K, alpha, lamb, Y, X, diff, ds)

    print(X[:3])
    print(Y[:3])