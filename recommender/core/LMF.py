import numpy as np
import os
import scipy.sparse as sps
import keras
from keras.backend import log, mean, exp
from keras.layers import Input, Embedding, Dot, Flatten, Add, Activation, Lambda
from keras.models import Model, load_model
from keras import regularizers, optimizers, initializers
from keras.utils import generic_utils
import tensorflow as tf
from typing import List
import matplotlib.pyplot as plt
from typing import Union
from sklearn.metrics import average_precision_score
from keras.callbacks import Callback
import pandas as pd

class LMFCallback(Callback):
    def __init__(self, Utrain: sps.csr_matrix, Utest: sps.csr_matrix, U: sps.csr_matrix, N: int, M: int):
        super(LMFCallback, self).__init__()
        self.Utrain = Utrain
        self.Utest = Utest
        self.U = U
        self.APs = []
        self.ep = []
        self.en = []
        self.epochs = []
        self.N = N
        self.M = M

    def on_epoch_end(self, epoch, logs=None):

        N = self.N
        M = self.M

        up, yp, rp = get_pos_ratings(self.Utest, np.arange(N), M)
        uup, nup = np.unique(up, return_counts=True)
        un, yn = get_neg_ratings(self.U, np.arange(N), M, samples_per_user=nup)
        rn = np.zeros(shape=un.shape, dtype=float)
        eval_result = self.model.model.evaluate(
            {'u_in': up, 'i_in': yp},
            {'phat': rp.astype(np.bool).astype(np.float, copy=True)}, batch_size=10000
        )
        self.ep.append(np.sqrt(eval_result[self.model.model.metrics_names.index('phat_mean_squared_error')]))
        eval_result = self.model.model.evaluate(
            {'u_in': un, 'i_in': yn},
            {'phat': rn.astype(np.bool).astype(np.float, copy=True)}, batch_size=10000
        )
        self.en.append(np.sqrt(eval_result[self.model.model.metrics_names.index('phat_mean_squared_error')]))

        u_test = np.concatenate([up, un])
        i_test = np.concatenate([yp, yn])
        r_test = np.concatenate([rp, rn]).astype(np.bool).astype(np.float, copy=True)
        phat, _, _, _ = self.model.predict(u_test, i_test)
        self.APs.append(average_precision_score(r_test, phat))
        self.epochs.append(epoch)

        print(pd.DataFrame({'epoch': self.epochs[-1:], 'AP': self.APs[-1:], 'EP': self.ep[-1:], 'EN':self.en[-1:]}))

    def save_result(self, outfile):
        pd.DataFrame({'epoch': self.epochs, 'AP': self.APs, 'EP': self.ep, 'EN':self.en}).to_csv(outfile, index=False)

def sample_neg(pos, M, s):
    neg = np.zeros(shape=(0,), dtype=int)
    while(neg.shape[0]==0):
        neg = np.random.choice(M, size=min(s,M), replace=False).astype(int)
        neg = neg[np.in1d(neg, pos, assume_unique=True, invert=True)]
        break
    return neg

def get_neg_ratings(R, users, M, samples_per_user:Union[np.ndarray, int]=50):

    ru = R[users, :]
    if type(samples_per_user) is int:
        samples_per_user = np.ones(shape=(users.shape[0],), dtype=int)*samples_per_user
    ns = np.sum(samples_per_user)
    up = np.zeros(ns, dtype=int)
    yp = np.zeros(ns, dtype=int)

    offs = 0
    for u, (user, s) in enumerate(zip(users, samples_per_user)):
        neg = sample_neg(ru[u].indices, M, s)
        up[offs: offs+neg.shape[0]] = user
        yp[offs: offs+neg.shape[0]] = neg
        offs += neg.shape[0]

    return up[:offs], yp[:offs]

def get_pos_ratings(R, users, M, batchsize=None):

    ru = R[users, :]
    l = np.max(ru.getnnz(axis=1))
    if batchsize is None:
        batchsize = len(users)
    up = np.zeros(shape=(batchsize*l), dtype=np.int)
    rp = np.zeros(shape=(batchsize*l), dtype=np.float)
    yp = M*np.ones(shape=(batchsize*l), dtype=np.int)
    offs = 0
    for u, user in enumerate(users):
        numy = ru[u].data.shape[0]
        up[offs: offs+numy] = user
        rp[offs: offs+numy] = ru[u].data
        yp[offs: offs+numy] = ru[u].indices
        offs += numy

    return up[:offs], yp[:offs], rp[:offs]

class LogisticMatrixFactorizer(object):

    def __init__(self, model_path, N, M, f=10, lr=0.01, lamb=0.01, alpha=40.0, bias=False, epochs=30, batchsize=5000, mode='train'):
        self.mode = mode
        self.initialize(model_path, N, M, f, lr, lamb, alpha=alpha, bias=bias, epochs=epochs, batchsize=batchsize)

    def initialize(self, model_path, N, M, f=10, lr=0.01, lamb=0.01, alpha=40.0, epochs=30, batchsize=5000, bias=False):

        self.model_path = model_path
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = lr

        if self.mode == 'predict':
            self.model = load_model(self.model_path, compile=True)
            self.model.summary()
            return

        self.M = M
        self.N = N

        u_in = Input(shape=(1,), dtype='int32', name='u_in')
        i_in = Input(shape=(1,), dtype='int32', name='i_in')

        X = Embedding(N, f, dtype='float32',
                      embeddings_regularizer=regularizers.l2(lamb), input_length=1,
                      embeddings_initializer=initializers.RandomNormal(seed=42), name='P')
        Y = Embedding(M, f, dtype='float32',
                      embeddings_regularizer=regularizers.l2(lamb), input_length=1,
                      embeddings_initializer=initializers.RandomNormal(seed=42), name='Q')

        x = X(u_in)
        y = Y(i_in)
        self.vars = {'X': X, 'Y': Y}

        if bias:
            # currently does not work with tf.function: https://github.com/keras-team/keras/issues /13671
            # seems like Multiply() and Add() have problems? TODO: try to recreate!
            Bu = Embedding(N, 1, dtype='float32', embeddings_initializer='random_normal', name='Bu')
            Bi = Embedding(M, 1, dtype='float32', embeddings_initializer='random_normal', name='Bi')
            bp = Bu(u_in)
            bq = Bi(i_in)
            z = Dot(2)([x, y])
            self.vars.update({'Bu': Bu, 'Bi': Bi})
            # Add(name='rhat')([Flatten()(Dot(2)([x, y])), bp, bq])
            # Need to do this because the add layer doesn't work in tf function
            rhat = Flatten(name='rhat')(Lambda(lambda x: x[0]+x[1]+x[2])([z, bp, bq]))
            phat = Activation('sigmoid', name='phat')(rhat)
        else:
            rhat = Flatten(name='rhat')(Dot(2)([x, y]))
            phat = Activation('sigmoid', name='phat')(rhat)

        model = Model(inputs=[u_in, i_in], outputs=[phat, rhat, x, y])

        loss_fn = PLMFLoss(alpha=alpha)

        self.model = model
        self.model.compile(
            optimizers.Adam(lr), loss={'phat': 'mean_squared_error'}, metrics={'phat': 'mean_squared_error'}
        )
        self.loss_fn = loss_fn

        self.model.summary(line_length=88)

        return

    def fit(self, Utrain, Utest, U, cb: Union[Callback, List[Callback]]=None): #, u_train, i_train, r_train, u_test, i_test, r_test):

        if cb is not None and type(cb)==Callback:
            cb = [cb]

        model = self.model
        X = self.vars['X']
        Y = self.vars['Y']
        Bu = self.vars.get('Bu')
        Bi = self.vars.get('Bi')
        loss_fn = self.loss_fn
        epochs = self.epochs

        num_seen = tf.Variable(shape=(), initial_value=0., dtype='float32')
        cur_batchsize = tf.Variable(shape=(), initial_value=0., dtype='float32')
        acc_loss = tf.Variable(shape=(), initial_value=0.)

        opt = tf.keras.optimizers.Adam(self.lr)

        Y.trainable = False
        if Bi is not None:
            Bi.trainable = False
        acc_grads_X = []
        for weight in model.trainable_weights:
            shape = weight.shape
            acc_grads_X.append(tf.Variable(shape=shape, initial_value=np.zeros(shape=shape, dtype='float32')))
        train_step_X = tf.function(experimental_relax_shapes=True)(train_step)

        Y.trainable = True
        X.trainable = False
        if Bu is not None:
            Bi.trainable = True
            Bu.trainable = False
        acc_grads_Y = []
        for weight in model.trainable_weights:
            shape = weight.shape
            acc_grads_Y.append(tf.Variable(shape=shape, initial_value=np.zeros(shape=shape, dtype='float32')))
        train_step_Y = tf.function(experimental_relax_shapes=True)(train_step)  # force retrace
        X.trainable = True

        N = X.input_dim
        M = Y.input_dim
        trace = np.zeros(shape=(epochs))

        for epoch in range(epochs):

            for phase in ['X', 'Y']:

                if phase == 'X':
                    X.trainable = True
                    Y.trainable = False
                    if Bu is not None:
                        Bi.trainable = False
                        Bu.trainable = True
                    acc_grads = acc_grads_X
                    train_step_fn = train_step_X
                else:
                    Y.trainable = True
                    X.trainable = False
                    if Bi is not None:
                        Bi.trainable = True
                        Bu.trainable = False
                    acc_grads = acc_grads_Y
                    train_step_fn = train_step_Y

                progbar = generic_utils.Progbar(N)

                for i in range(0, N, self.batchsize):
                    up, yp, rp = get_pos_ratings(Utrain, np.arange(i, min(i + self.batchsize, N)), M)
                    uup, nup = np.unique(up, return_counts=True)
                    un, yn = get_neg_ratings(U, np.arange(i, min(i+self.batchsize, N)), M, samples_per_user=nup)
                    rn = np.zeros(shape=un.shape, dtype=float)
                    x = tf.constant(np.expand_dims(np.concatenate([up, un]), axis=1), tf.int32)
                    y = tf.constant(np.expand_dims(np.concatenate([yp, yn]), axis=1), tf.int32)
                    z = tf.constant(np.expand_dims(np.concatenate([rp, rn]), axis=1), tf.float32)
                    cur_batchsize.assign(float(x.shape[0]))
                    train_step_fn(model, x, y, z, loss_fn, acc_grads, acc_loss, num_seen, cur_batchsize)
                    num_seen.assign_add(cur_batchsize)
                    progbar.add(min(self.batchsize, N-i), values=[(phase + ' loss', acc_loss)])

                trace[epoch] += acc_loss
                num_seen.assign(0.)
                acc_loss.assign(0.)

                opt.apply_gradients(zip(acc_grads, model.trainable_weights))
                for acc_grad in acc_grads:
                    acc_grad.assign(tf.zeros(shape=acc_grad.shape, dtype=acc_grad.dtype))

            if cb is not None:
                for c in cb:
                    c.on_epoch_end(epoch)
            self.save_as_epoch(epoch)

        return trace

    def save(self, model_path):
        self.model.save(model_path)

    def save_as_epoch(self, epoch:Union[str,int]='last'):
        model_name = 'model-{:03d}.h5' if type(epoch) == int else 'model-{:s}.h5'
        self.model.save(os.path.join(self.model_path, model_name.format(epoch)))

    def predict(self, u, i):
        return self.model.predict({'u_in': u, 'i_in': i}, batch_size=10000)

class PLMFLoss(keras.losses.Loss):
    """
    Args:
      alpha: PLMF alpha scaling of positive examples
    """
    def __init__(self, alpha,
                 name='plmf_loss'):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, r_ui, rhat):
        alpha = self.alpha
        return mean(-alpha * r_ui * rhat + (1 + alpha * r_ui) * log(1 + exp(rhat)))

# @tf.function(experimental_relax_shapes=True)
def train_step(
        model: Model, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor,
        loss_fn: PLMFLoss, acc_grads: List[tf.Tensor], acc_loss: tf.Tensor, N: tf.Tensor, M: tf.Tensor):

    print("tracing tf function graph...")

    with tf.GradientTape() as tape:
        _, rhat, _, _ = model([x, y])
        loss_value = loss_fn(z, rhat)

    grads = tape.gradient(loss_value, model.trainable_weights)

    for acc_grad, grad in zip(acc_grads, grads):
        acc_grad.assign_add(grad)

    acc_loss.assign_add(loss_value)

    return loss_value, grads

if __name__=="__main__":

    M = 800
    N = 350
    P = 1500
    k = 10
    alpha = 5.0
    lamb = 0.1
    np.random.seed(42)
    data = np.random.uniform(1, 10, size=(P,))
    c = np.random.randint(0, M * N, size=(P,))
    c = np.unique(c)
    data = data[:len(c)]
    rows = c // M
    cols = c % M
    split = int(0.8 * len(data))

    U = sps.csr_matrix((data, (rows, cols)), shape=(N, M))
    Utrain = sps.csr_matrix((data[:split], (rows[:split], cols[:split])), shape=(N, M))
    Utest = sps.csr_matrix((data[split:], (rows[split:], cols[split:])), shape=(N, M))

    model_folder = 'D:\\PycharmProjects\\recommender\\models'
    save_path = os.path.join(model_folder, "LMF_tmp")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lmf = LogisticMatrixFactorizer(
        save_path, N, M, f=k, lr=0.1,
        alpha=alpha, bias=False, epochs=100, batchsize=50
    )


    rows = rows.astype(np.int32)
    cols = cols.astype(np.int32)
    data = data.astype(np.float32)
    trace = lmf.fit(Utrain, Utest)

    plt.plot(trace)
    plt.show()