from abc import ABC, abstractmethod
from .Environment import RecAlgo, Environment, Algorithm
import numpy as np
from ..data.iterators import AddBias, AddRatedItems, link, Normalizer

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
        add_rated_items = AddRatedItems(U)
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