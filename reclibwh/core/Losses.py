import keras
from keras.backend import log, mean, exp
import tensorflow as tf

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
        r_ui = tf.cast(r_ui, tf.float32)
        return mean(-alpha * r_ui * rhat + (1 + alpha * r_ui) * log(1 + exp(rhat)))
