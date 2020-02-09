import numpy as np
import os
import scipy.sparse as sps
import keras
from keras.backend import log, mean, exp
from keras.layers import Input, Embedding, Dot, Flatten, Add, Activation
from keras.models import Model, load_model
from keras import regularizers, optimizers, initializers
from keras.utils import generic_utils
import tensorflow as tf
from typing import List
from tqdm import tqdm

def sample_neg(pos, M, s):
    neg = np.zeros(shape=(0,), dtype=int)
    while(neg.shape[0]==0):
        neg = np.random.choice(M, size=s, replace=False).astype(int)
        neg = neg[np.in1d(neg, pos, assume_unique=True, invert=True)]
        break
    return neg

def get_neg_ratings(R, users, M, samples_per_user=50):

    ru = R[users, :]
    s = samples_per_user
    up = np.zeros(len(users)*s, dtype=int)
    yp = np.zeros(len(users) * s, dtype=int)

    offs = 0
    for u, user in enumerate(users):
        neg = sample_neg(ru[u].indices, M, s)
        up[offs: offs+neg.shape[0]] = user
        yp[offs: offs+neg.shape[0]] = neg
        offs += neg.shape[0]

    return up[:offs], yp[:offs]

class LogisticMatrixFactorizer(object):

    def __init__(self, model_dir, N, M, f=10, lr=0.01, lamb=0.01, alpha=40.0, bias=False, epochs=30, batchsize=5000, mode='train'):
        self.mode = mode
        self.initialize(model_dir, N, M, f, lr, lamb, alpha=alpha, bias=bias, epochs=epochs, batchsize=batchsize)

    def initialize(self, model_path, N, M, f=10, lr=0.01, lamb=0.01, alpha=40.0, epochs=30, batchsize=5000, bias=False):

        self.model_path = model_path
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = lr

        if self.mode == 'predict':
            self.model = load_model(self.model_path, compile=True)
            self.model.summary()
            return

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
            bp = Flatten()(Bu(u_in))
            bq = Flatten()(Bi(i_in))
            self.vars.update({'Bu': Bu, 'Bi': Bi})
            rhat = Add(name='rhat')([Flatten()(Dot(2)([x, y])), bp, bq])
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

    def fit(self, u_train, i_train, r_train, u_test, i_test, r_test):

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (np.expand_dims(u_train, axis=1), np.expand_dims(i_train, axis=1), np.expand_dims(r_train, axis=1).astype('float32'))
        ).batch(self.batchsize)

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

        opt = tf.keras.optimizers.Adam(0.1)

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

                progbar = generic_utils.Progbar(np.ceil(float(len(u_train)) / self.batchsize))

                for x, y, z in tqdm(train_dataset):

                    cur_batchsize.assign(float(len(x)))
                    train_step_fn(model, x, y, z, loss_fn, acc_grads, acc_loss, num_seen, cur_batchsize)
                    num_seen.assign_add(cur_batchsize)
                    progbar.add(1, values=[(phase + ' loss', acc_loss)])

                num_seen.assign(0.)

                acc_loss.assign(0.)
                opt.apply_gradients(zip(acc_grads, model.trainable_weights))
                for acc_grad in acc_grads:
                    acc_grad.assign(tf.zeros(shape=acc_grad.shape, dtype=acc_grad.dtype))

        #Evaluate the model
        phat, _, _, _ = self.model.predict([u_test, i_test])
        eval_result = self.model.evaluate(
            {'u_in': u_test, 'i_in': i_test}, {'phat': r_test.astype(np.bool).astype(np.float, copy=True)}, batch_size=10000
        )
        print('Test: RMSE: %f ' % np.sqrt(eval_result[self.model.metrics_names.index('phat_mean_squared_error')]))

    def save(self):
        self.model.save(os.path.join(self.model_path, 'model.h5'))

    def predict(self, u, i):
        return self.model.predict({'u_in': u, 'i_in': i})

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
def train_step(model: Model, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor, loss_fn: PLMFLoss, acc_grads: List[tf.Tensor], acc_loss: tf.Tensor, N: tf.Tensor, M: tf.Tensor):

    with tf.GradientTape() as tape:
        _, rhat, _, _ = model([x, y])
        loss_value = loss_fn(z, rhat)

    grads = tape.gradient(loss_value, model.trainable_weights)

    for acc_grad, grad in zip(acc_grads, grads):
        acc_grad_new = acc_grad*(N/(N + M)) + grad*(M/(N + M))
        acc_grad.assign_add(acc_grad_new)

    acc_loss_new = acc_loss*(N/(N + M)) + loss_value*(M/(N + M))
    acc_loss.assign(acc_loss_new)

    return loss_value, grads

if __name__=="__main__":

    M = 800
    N = 350
    P = 1500
    k = 10
    alpha = 10.0
    lamb = 0.1
    np.random.seed(42)
    data = np.random.uniform(1, 10, size=(P,))
    c = np.random.randint(0, M * N, size=(P,))
    c = np.unique(c)
    data = data[:len(c)]
    rows = c // M
    cols = c % M

    U = sps.csr_matrix((data, (rows, cols)), shape=(N, M))

    u_neg, i_neg = get_neg_ratings(U, np.arange(N), M, samples_per_user=5)
    n_neg = u_neg.shape[0]

    data = np.concatenate([data, np.zeros(n_neg)])#[:125]
    rows = np.concatenate([rows, u_neg])#[:125]
    cols = np.concatenate([cols, i_neg])#[:125]

    model_folder = 'D:\\PycharmProjects\\recommender\\models'
    save_path = os.path.join(model_folder, "LMF_tmp")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lmf = LogisticMatrixFactorizer(save_path, N, M, f=k, lr=0.1, alpha=alpha, bias=False, epochs=10, batchsize=100)
    split = int(0.8*len(data))
    rows = rows.astype(np.int32)
    cols = cols.astype(np.int32)
    data = data.astype(np.float32)
    lmf.fit(
        rows[:split], cols[:split], data[:split],
        rows[split:], cols[split:], data[split:]
    )