from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense, Activation, Layer
from tensorflow.keras.losses import Loss, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.initializers import Constant
import tensorflow as tf
import scipy.sparse as sps
import numpy as np
from ..utils.ItemMetadata import ExplicitDataFromCSV
from datetime import datetime

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.training import _minimize
from tensorflow.python.eager import backprop

class RegularizerInspector(Callback):

    def __init__(self, regularizer_layers=None):
        super(RegularizerInspector, self).__init__()
        self.batch_count = 0
        self.regularizer_layers = regularizer_layers

    def on_epoch_end(self, epoch, logs):
        # simple printout
        for regularizer_layer in self.regularizer_layers:
            lamb = self.model.get_layer(regularizer_layer)
            print(regularizer_layer, lamb.get_weights()[0])
    
    def on_batch_end(self, batch, logs):
        # log to tensorboard (if available)
        self.batch_count += 1
        for regularizer_layer in self.regularizer_layers:
            lamb = self.model.get_layer(regularizer_layer)
            with tf.name_scope(regularizer_layer):
                tf.summary.scalar('log_lambda', tf.reduce_mean(lamb.get_weights()[0]), step=self.batch_count)

class AdaptiveRegularizer(Layer):

    def __init__(
        self, N, D, Nranked, initial_value=0., alpha=0.01, beta=1e-07, **kwargs
    ):
        super(AdaptiveRegularizer, self).__init__(**kwargs)
        self.initial_value = initial_value
        self.N = N
        self.D = D
        assert alpha > 0., "alpha needs to be a positive float"
        self.Nranked = Nranked
        self.alpha = alpha
        self.mult = alpha*(N*D)/Nranked
        self.beta = beta
    
    def build(self, input_shape):
        self.log_lambda = self.add_weight(shape=(1,), initializer=Constant(self.initial_value), trainable=True, name='log_lambda')
    
    def call(self, x):

        log_lambda = self.log_lambda
        beta = self.beta
        mult = self.mult
        lamb = tf.squeeze(tf.exp(log_lambda))
        batchsize = tf.shape(x)[0]
        reg_loss = lamb*(tf.reduce_sum(tf.square(x)) + 2*beta) - mult*float(batchsize)*tf.squeeze(log_lambda)
        self.add_loss(reg_loss)
        return reg_loss
    
    def get_config(self):
        return {
            'N': self.N, 'D': self.D, 'Nranked': self.Nranked, 'name': self.name,
            'alpha': self.alpha, 'beta': self.beta, 'initial_value': self.initial_value
        }
