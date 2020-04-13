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
from typing import Optional, Set, Iterable
from recommender.utils.utils import get_pos_ratings, get_neg_ratings
import json

class LMFCallback(Callback):
    def __init__(self, Utrain: sps.csr_matrix, Utest: sps.csr_matrix, U: sps.csr_matrix, N: int, M: int, users: Optional[Iterable[int]]=None):
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
        if users is not None: self.users = users
        else: self.users = np.arange(N)

    def on_epoch_end(self, epoch, logs=None):

        N = self.N
        M = self.M
        users = self.users

        up, yp, rp = get_pos_ratings(self.Utest, users, M)
        uup, nup = np.unique(up, return_counts=True)
        un, yn = get_neg_ratings(self.U, users, M, samples_per_user=nup)
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

class LogisticMatrixFactorizer(object):

    def __init__(self, model_path, N, M, f=10, lr=0.01, lamb=0.01, alpha=40.0, bias=False, epochs=30, batchsize=5000, mode='train', saved_model=None):
        self.mode = mode
        self.initialize(model_path, N, M, f, lr, lamb, alpha=alpha, bias=bias, epochs=epochs, batchsize=batchsize, saved_model=saved_model)

    @staticmethod
    def make_model(N, M, f=10, lamb=0.01, bias=False):

        u_in = tf.keras.layers.Input(shape=(1,), dtype='int32', name='u_in')
        i_in = tf.keras.layers.Input(shape=(1,), dtype='int32', name='i_in')


        X = tf.keras.layers.Embedding(N, f, dtype='float32',
                      embeddings_regularizer=tf.keras.regularizers.l2(lamb), input_length=1,
                      embeddings_initializer=tf.keras.initializers.RandomNormal(seed=42), name='P')
        Y = tf.keras.layers.Embedding(M, f, dtype='float32',
                      embeddings_regularizer=tf.keras.regularizers.l2(lamb), input_length=1,
                      embeddings_initializer=tf.keras.initializers.RandomNormal(seed=42), name='Q')

        x = X(u_in)
        y = Y(i_in)

        if bias:
            # currently does not work with tf.function: https://github.com/keras-team/keras/issues /13671
            # seems like Multiply() and Add() have problems? TODO: try to recreate!
            Bu = tf.keras.layers.Embedding(N, 1, dtype='float32', embeddings_initializer='random_normal', name='Bu')
            Bi = tf.keras.layers.Embedding(M, 1, dtype='float32', embeddings_initializer='random_normal', name='Bi')
            bp = Bu(u_in)
            bq = Bi(i_in)
            z = tf.keras.layers.Dot(2)([x, y])
            # Add(name='rhat')([Flatten()(Dot(2)([x, y])), bp, bq])
            # Need to do this because the add layer doesn't work in tf function
            rhat = tf.keras.layers.Flatten(name='rhat')(tf.keras.layers.Lambda(lambda x: x[0] + x[1] + x[2])([z, bp, bq]))
            phat = tf.keras.layers.Activation('sigmoid', name='phat')(rhat)
        else:
            rhat = tf.keras.layers.Flatten(name='rhat')(tf.keras.layers.Dot(2)([x, y]))
            phat = tf.keras.layers.Activation('sigmoid', name='phat')(rhat)

        model = tf.keras.Model(inputs=[u_in, i_in], outputs=[phat, rhat, x, y])

        model.compile(
            tf.keras.optimizers.Adam(0.1), loss={'phat': 'mean_squared_error'}, metrics={'phat': 'mean_squared_error'}
        )
        return model

    @property
    def vars(self):
        if self.model is None:
            return {}
        vars = {'X': self.model.get_layer('P'), 'Y': self.model.get_layer('Q')}
        try:
            vars.update({'Bu': self.model.get_layer('Bu'), 'Bi': self.model.get_layer('Bi'), })
        except ValueError:
            print("no bias detected for model")
        return vars

    def initialize(
            self, model_path, N, M, f=10, lr=0.01, lamb=0.01, alpha=40.0, epochs=30, batchsize=5000, bias=False, saved_model=None
    ):

        self.saved_model = saved_model
        self.model_path = model_path
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = lr

        self.model_kwargs = {'N':N, 'M':M, 'f':f, 'lamb': lamb, 'bias':bias}

        if saved_model is not None: self.model = tf.keras.models.load_model(self.saved_model, compile=True)
        else: self.model = LogisticMatrixFactorizer.make_model(**self.model_kwargs)

        loss_fn = PLMFLoss(alpha=alpha)
        self.loss_fn = loss_fn
        self.model.summary(line_length=88)
        return

    def _add_users(self, num=1):

        self.model_kwargs['N'] += num
        # update old embeddings
        new_model= LogisticMatrixFactorizer.make_model(**self.model_kwargs)
        oldX = self.model.get_layer('P').get_weights()[0]
        newX = np.concatenate([oldX, np.random.normal(0,1.,size=(num, self.model_kwargs['f']))])
        new_model.get_layer('P').set_weights([newX])
        new_model.get_layer('Q').set_weights(self.model.get_layer('Q').get_weights())
        # update old biases
        if self.model_kwargs['bias']:
            oldBu = self.model.get_layer('Bu').get_weights()[0]
            newBu = np.concatenate([oldBu, np.random.normal(0, 1., size=(num, 1))])
            new_model.get_layer('Bu').set_weights([newBu])
            new_model.get_layer('Bi').set_weights(self.model.get_layer('Bi').get_weights())
        self.model = new_model
        self.model.summary()

    def fit(self, Utrain, Utest, U, users:Optional[np.array]=None, cb: Union[Callback, List[Callback]]=None,
            exclude_phase:Optional[Set]=None, ckpt_json:Optional[str]=True): #, u_train, i_train, r_train, u_test, i_test, r_test):

        model = self.model
        X = self.vars['X']
        Y = self.vars['Y']
        Bu = self.vars.get('Bu')
        Bi = self.vars.get('Bi')
        loss_fn = self.loss_fn
        epochs = self.epochs
        batchsize=self.batchsize

        N = X.input_dim
        M = Y.input_dim

        if cb is not None and type(cb)==Callback: cb = [cb]
        if users is None: users = np.arange(0, N)

        num_seen = tf.Variable(initial_value=0., dtype='float32')
        cur_batchsize = tf.Variable(initial_value=0., dtype='float32')
        acc_loss = tf.Variable(initial_value=0.)

        opt = tf.keras.optimizers.Adam(0.1)
        trace = None
        start_epoch = -1
        tf_manager = None
        if ckpt_json is not None: start_epoch, trace, tf_ckpt, tf_manager = self.load_ckpt_json(ckpt_json, opt, model)
        if trace is None: trace = np.zeros(shape=(epochs))

        phaseVariables = {
            'X': {'vars': [X,Bu], 'acc_grads': [], 'train_step_fn': None},
            'Y': {'vars': [Y,Bi], 'acc_grads': [], 'train_step_fn': None},
        }

        for p, stuff_p in phaseVariables.items():
            stuff_p['vars'] =  list(filter(lambda x: x is not None, stuff_p['vars']))

        for p, stuff_p in phaseVariables.items():
            for q, stuff_q in phaseVariables.items():
                for v in stuff_q['vars']: v.trainable=False
            for v in stuff_p['vars']: v.trainable = True
            acc_grads = []
            for weight in model.trainable_weights:
                shape = weight.shape
                acc_grads.append(tf.Variable(shape=shape, initial_value=np.zeros(shape=shape, dtype='float32')))
            stuff_p['acc_grads'] = acc_grads
            stuff_p['train_step_fn'] = tf.function(experimental_relax_shapes=True)(train_step) # for retrace

        if cb is not None:
            for c in cb: c.on_epoch_end(-1)

        for epoch in range(start_epoch+1,epochs):

            for phase in ['X', 'Y']:

                if exclude_phase is not None:
                    if phase in exclude_phase: continue

                for p, stuff in phaseVariables.items():
                    for v in stuff['vars']: v.trainable = False
                stuff = phaseVariables[phase]
                for v in stuff['vars']: v.trainable = True
                acc_grads = stuff['acc_grads']
                train_step_fn = stuff['train_step_fn']  # for retrace

                progbar = generic_utils.Progbar(len(users))

                for i in range(0, len(users), batchsize):
                    us = users[i:min(i + batchsize, len(users))]  # np.arange(i, min(i + self.batchsize, N))
                    up, yp, rp = get_pos_ratings(Utrain, us, M)
                    uup, nup = np.unique(up, return_counts=True)
                    un, yn = get_neg_ratings(U, us, M, samples_per_user=nup)
                    rn = np.zeros(shape=un.shape, dtype=float)
                    x = tf.constant(np.expand_dims(np.concatenate([up, un]), axis=1), tf.int32)
                    y = tf.constant(np.expand_dims(np.concatenate([yp, yn]), axis=1), tf.int32)
                    z = tf.constant(np.expand_dims(np.concatenate([rp, rn]), axis=1), tf.float32)
                    cur_batchsize.assign(float(x.shape[0]))
                    train_step_fn(model, x, y, z, loss_fn, acc_grads, acc_loss, num_seen, cur_batchsize)
                    num_seen.assign_add(cur_batchsize)
                    progbar.add(min(self.batchsize, len(users)-i), values=[(phase + ' loss', acc_loss)])

                trace[epoch] += acc_loss
                num_seen.assign(0.)
                acc_loss.assign(0.)

                opt.apply_gradients(zip(acc_grads, model.trainable_weights))
                for acc_grad in acc_grads:
                    acc_grad.assign(tf.zeros(shape=acc_grad.shape, dtype=acc_grad.dtype))

            if cb is not None:
                for c in cb: c.on_epoch_end(epoch)
            if ckpt_json is not None: self.save_ckpt(ckpt_json, epoch, trace, tf_manager)

        return trace

    def save(self, model_path):
        tf.keras.models.save_model(self.model, model_path, include_optimizer=True)

    def save_as_epoch(self, epoch:Union[str,int]='last'):
        model_name = 'model-{:03d}.h5' if type(epoch) == int else 'model-{:s}.h5'
        save_path = os.path.join(self.model_path, model_name.format(epoch))
        self.save(save_path)
        return

    def save_trace(self, trace):
        save_path = os.path.join(self.model_path, 'trace.csv')
        pd.DataFrame({
            'epoch': list(range(len(trace))), 'trace': trace,
        }).to_csv(save_path, index=False)
        return save_path

    def load_trace(self, trace_path):
        return np.array(pd.read_csv(trace_path)['trace'])

    def load_ckpt_json(self, ckpt_json, opt: tf.keras.optimizers.Optimizer, model: tf.keras.models.Model):

        tf_ckpt_dir = os.path.join(self.model_path, 'tf_ckpts')
        tf_ckpt_path = None
        trace = None
        start_epoch = -1

        if os.path.exists(ckpt_json):
            with open(ckpt_json, 'r') as fp: ckpt = json.load(fp)
            assert 'epoch' in ckpt and 'trace_path' in ckpt, "model_path, epoch, and trace_path required in ckpt. found {}".format(
                ckpt.keys())
            trace = self.load_trace(ckpt['trace_path'])
            start_epoch = ckpt['epoch']
            tf_ckpt_path = ckpt['tf_ckpt_path']
            tf_ckpt_dir = os.path.dirname(tf_ckpt_path)

        tf_ckpt = tf.train.Checkpoint(optimizer=opt, model=model)
        tf_manager = tf.train.CheckpointManager(tf_ckpt, tf_ckpt_dir, max_to_keep=3)
        if tf_ckpt_path is None: tf_ckpt_path = tf_manager.latest_checkpoint
        tf_ckpt.restore(tf_ckpt_path)
        if tf_ckpt_path: print("Restored from {}".format(tf_manager.latest_checkpoint))
        else: print("Initializing from scratch.")

        return start_epoch, trace, tf_ckpt, tf_manager

    def save_ckpt(self, ckpt_json, epoch, trace, tf_manager: tf.train.CheckpointManager):
        model_path = self.save_as_epoch(epoch)
        trace_path = self.save_trace(trace)
        tf_ckpt_path = tf_manager.save()
        with open(ckpt_json, 'w') as fp:
            json.dump({'model_path': model_path, 'trace_path': trace_path, 'epoch': epoch, 'tf_ckpt_path': tf_ckpt_path}, fp, indent=4)

    def predict(self, u, i):
        return self.model.predict({'u_in': u, 'i_in': i}, batch_size=50000)

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
        model: tf.keras.models.Model, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor,
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