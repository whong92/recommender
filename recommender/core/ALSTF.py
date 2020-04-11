import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
import tensorflow as tf

from recommender.core.ALS import ALSTF, run_single_step, sparse2dataset
from .ALS import ALS

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

    trace = als.train(R)
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