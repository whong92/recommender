from abc import ABC, abstractmethod
from .Environment import RecAlgo, Environment, Algorithm, SimAlgo
import numpy as np
from ..data.iterators import AddBias, AddRatedItems, link, Normalizer
import tensorflow as tf

class SimpleMFRecAlgo(RecAlgo):

    """
    requires that the encapsulating environemtn has an 'algo' object that implements
    the Algorithm interface (with a predict method)
    """

    def __init__(self, env: Environment, algo: Algorithm, output_key='rhat'):

        self.__env = env
        self.__output_key = output_key
        self.__algo = algo

    def recommend(self, user):
        M = self.__env.get_state()['data_conf']['M']
        u_in = np.array(user)
        # manual prediction by enumerating all stuff
        nu = u_in.shape[0]
        ni = M
        rhats = self.__algo.predict(
            u_in=np.repeat(np.expand_dims(u_in, 0), ni).transpose().flatten(),
            i_in=np.tile(np.arange(ni), nu)
        )[self.__output_key]
        rhats = rhats.reshape(nu, -1)

        return np.argsort(rhats)[:, ::-1], np.sort(rhats)[:, ::-1]

@tf.function
def replacenan(t):
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)

@tf.function
def cosine_similarity_tf(x: tf.Tensor, y: tf.Tensor):
    """
    :param x: N x k tensor
    :param y: M x k tensor
    :return: N x M similarity tensor
    """
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=0)
    x, _ = tf.linalg.normalize(x, axis=-1, ord=2)
    y, _ = tf.linalg.normalize(y, axis=-1, ord=2)
    x = replacenan(x)
    y = replacenan(y)
    return tf.reduce_sum(tf.multiply(x, y), axis=-1)

class SimpleMFSimAlgo(SimAlgo):

    def __init__(self, env: Environment, output_emb='X', model_num=0):

        self.__env = env
        self.__output_emb = output_emb
        self.__model_num = model_num

    def similar(self, item: np.array):

        self.__model = self.__env['model'][self.__model_num]
        M = self.__env.get_state()['data_conf']['M']

        assert item.ndim == 1
        item = np.expand_dims(item, axis=0)
        X = self.__model.get_layer(self.__output_emb).trainable_weights[0]
        x = tf.gather_nd(X, tf.convert_to_tensor(item))
        sims = cosine_similarity_tf(x, X)
        sims = sims.numpy()[:,:M]

        return np.argsort(sims)[:, ::-1], np.sort(sims)[:, ::-1]

class MFAsymRecAlgo(RecAlgo):

    def __init__(self, env: Environment, algo: Algorithm, output_key='rhat'):

        self.__env = env
        self.__output_key = output_key
        self.__algo = algo

    def recommend(self, user):
        M = self.__env.get_state()['data_conf']['M']
        u_in = np.array(user)
        # manual prediction by enumerating all stuff
        nu = u_in.shape[0]
        ni = M
        data = self.__env.get_state()['data']['mf_asym_rec_data']
        U = data['U']
        norm = data.get('norm')

        u_in = np.repeat(np.expand_dims(u_in, 0), ni).transpose().flatten()
        i_in = np.tile(np.arange(ni), nu)

        d = {'user': u_in}
        add_rated_items = AddRatedItems(U, item_key=None)
        add_bias = AddBias(U, item_key='user_rated_items', pad_val=-1)
        nit = Normalizer({} if not norm else {'user_rated_ratings': norm, 'bias': norm})
        row = None
        for r in link([[d], add_rated_items, add_bias, nit]): row = r
        uj_in = row['user_rated_items'] + 1
        rj_in = row['user_rated_ratings']
        bj_in = row['bias']

        # TODO: FIX!!!!!!
        rhats = self.__algo.predict(
            u_in=u_in, i_in=i_in, uj_in=uj_in, bj_in=bj_in, rj_in=rj_in
        )[self.__output_key]
        rhats = rhats.reshape(nu, -1)

        return np.argsort(rhats)[:, ::-1], np.sort(rhats)[:, ::-1]