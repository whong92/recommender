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
from tensorflow.python.eager import backprop

@tf.keras.utils.register_keras_serializable(package='Custom', name='PMFLoss')
def PMFLoss(r_ui, rhat):
    return tf.reduce_sum(tf.square(r_ui-rhat))

class RegularizerInspector(Callback):

    def __init__(self, regularizer_layers=None):
        super(RegularizerInspector, self).__init__()
        self.batch_count = 0
        self.regularizer_layers = regularizer_layers

    def on_epoch_end(self, epoch, logs=None):
        # simple printout
        for regularizer_layer in self.regularizer_layers:
            lamb = self.model.get_layer(regularizer_layer)
            print(regularizer_layer, lamb.get_weights()[0])
    
    def on_batch_end(self, batch, logs=None):
        # log to tensorboard (if available)
        self.batch_count += 1
        for regularizer_layer in self.regularizer_layers:
            lamb = self.model.get_layer(regularizer_layer)
            with tf.name_scope(regularizer_layer):
                tf.summary.scalar('log_lambda', tf.reduce_mean(lamb.get_weights()[0]), step=self.batch_count)

@tf.keras.utils.register_keras_serializable(package='Custom', name='AdaptiveRegularizer')
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
    
    def call(self, x, **kwargs):

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

@tf.keras.utils.register_keras_serializable(package='Custom', name='ReductionLayer')
class ReductionLayer(Layer):

    def __init__(self, method, alpha: float):

        super(ReductionLayer, self).__init__()
        assert method in {'mask_mean', 'mask_sum'}
        self.method = method
        self.alpha = alpha
    
    def build(self, input_shape):
        assert len(input_shape) == 3, "embeddings have to be of shape batchsize x input length x dimensionality"

    def call(self, x: tf.Tensor, mask: tf.Tensor=None, weights:tf.Tensor=None):
        """[summary]

        Arguments:
            x {tf.Tensor} -- [batch x input_length sized input]

        Keyword Arguments:
            mask {[type]} -- [The mask over input sequence] (default: {None})
            weights {[type]} -- [The weights over input sequence] (default: {None})

        Returns:
            [tf.Tensor] -- [the computed reduced embedding]
        """
        counts = tf.ones(shape=(1,))
        mask = tf.cast(mask, tf.float32)
        if (mask is not None) and (self.method in {'mask_mean'}):
            counts = tf.reduce_sum(mask, axis=[1], keepdims=True)
            counts = tf.math.pow(counts, self.alpha)
            # doesn't matter if counts is zero, if it is, the output embedding with be zero
            counts = tf.clip_by_value(counts, 1., float("inf")) 
        # mask out irrlevant values
        mask = tf.expand_dims(tf.multiply(mask, weights), axis=2)
        x = tf.multiply(mask, x)
        x = tf.divide(tf.reduce_sum(x, axis=1, keepdims=True), tf.expand_dims(counts, axis=2))
        return x
    
    def get_config(self):
        return {
            'method': self.method, 'alpha': self.alpha
        }

@tf.keras.utils.register_keras_serializable(package='Custom', name='ReducedEmbedding')
class ReducedEmbedding(Layer):

    def __init__(self, N, f, method, alpha=0.5, **kwargs):

        super(ReducedEmbedding, self).__init__(name=kwargs['name'])
        kwargs.pop('mask_zero', None)
        self.N = N
        self.f = f
        self.method = method
        self.alpha = alpha
        self.kwargs = kwargs
        self.E = Embedding(N, f, **kwargs, mask_zero=True)
        self.A = ReductionLayer(method, alpha=alpha)

    def call(self, x: tf.Tensor, w: tf.Tensor):
        """[An embedding layer that reduces the embedding along the input length dimension, giving a single reduced  embedding]

        Arguments:
            x {[tf.Tensor]} -- [batch x input_length sized input]
        """
        x = self.E(x)
        y = self.A(x, weights=w)
        return y
    

    def get_config(self):
        return {
            'N': self.N,  'f': self.f, 'method': self.method, 'alpha': self.alpha,
            **self.kwargs
        }


if __name__=="__main__":

    x = Input(shape=(4,))
    e = ReducedEmbedding(5, 5, 'mask_mean', alpha=0.5)(x)
    
    m = Model(inputs=x, outputs=e)
    print(m.predict(
        np.array([
            [0,0,1,2,0,1]
        ])
    ))