import tensorflow as tf
from ..utils.utils import csv2df, makeTfDataset, splitDf
import numpy as np
from ..recommenderInterface import Recommender

class RecommenderMF(Recommender):
    def __init__(self, model_file=None):
        super(RecommenderMF, self).__init__(model_file)
        self.model = None
        self.train = None
        self.test = None

    def _make_model(self, u_in, i_in):

        with tf.variable_scope("MF", reuse=tf.AUTO_REUSE):
            m = {}
            # model
            m.P = tf.get_variable("P", [N, f], dtype=tf.float64)
            m.Q = tf.get_variable("Q", [M, f], dtype=tf.float64)
            m.Bu = tf.get_variable("Bu", [N, ], dtype=tf.float64)
            m.Bi = tf.get_variable("Bi", [M, ], dtype=tf.float64)

            m.p = tf.nn.embedding_lookup(P, u_in)
            m.q = tf.nn.embedding_lookup(Q, i_in)
            m.bu = tf.nn.embedding_lookup(Bu, u_in)
            m.bi = tf.nn.embedding_lookup(Bi, i_in)
            m.rhat = tf.reduce_sum(tf.multiply(p, q), axis=1) + bu + bi

            self.model = m

    def _make_training_network(self):

        t = {}

        with tf.variable_scope("MF", reuse=tf.AUTO_REUSE):
            # input placeholders
            t.user = tf.placeholder(tf.int32, [None, ])
            t.item = tf.placeholder(tf.int32, [None, ])
            t.rating = tf.placeholder(tf.int32, [None, ])

            # training input
            t.user_it, t.u_in = makeTfDataset(user_in, batchsize, numepochs)
            t.item_it, t.i_in = makeTfDataset(item_in, batchsize, numepochs)
            t.rating_it, t.r_in = makeTfDataset(rate_in, batchsize, numepochs)

        m = self.model
        self._make_model(t.u_in, t.i_in)
        t.rmse = tf.losses.mean_squared_error(t.r_in, m.rhat)
        t.reg = tf.nn.l2_loss(m.p) + tf.nn.l2_loss(m.q) + tf.nn.l2_loss(m.bu) + tf.nn.l2_loss(m.bi)
        t.loss = tf.cast(t.rmse, tf.float64) + tf.reduce_mean(lamb * t.reg)
        t.opt = tf.train.RMSPropOptimizer(lr, decay).minimize(t.loss)


    def train(self, precompute_ud=False):
        pass

    def predict(self):
        pass

if __name__=="__main__":
    train_test_split = 0.8
    df, N, M = csv2df('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv', 'userId', 'movieId', 'rating')
    D_train, D_test = splitDf(df, train_test_split)

    Users_train = D_train['user']
    Items_train = D_train['item']
    Ratings_train = D_train['rating']

    Users_test = D_test['user']
    Items_test = D_test['item']
    Ratings_test = D_test['rating']

    print_every=500

    f = 10 # latent factor dimensionality
    lamb = 0.01
    batchsize = 32
    numepochs = 10
    lr = 0.01
    decay = 0.0
    mu = np.mean(np.array(Ratings_train))
    model_vars = {}

    with tf.variable_scope("MF", reuse=tf.AUTO_REUSE):

        # input placeholders
        user_in = tf.placeholder(tf.int32, [None,])
        item_in = tf.placeholder(tf.int32, [None,])
        rate_in = tf.placeholder(tf.int32, [None,])
        # training input
        user_iterator, u_in = makeTfDataset(user_in, batchsize, numepochs)
        item_iterator, i_in = makeTfDataset(item_in, batchsize, numepochs)
        target_iterator, r_in = makeTfDataset(rate_in, batchsize, numepochs)

        # model
        P = tf.get_variable("P", [N, f], dtype=tf.float64)
        Q = tf.get_variable("Q", [M, f], dtype=tf.float64)
        Bu = tf.get_variable("Bu", [N, ], dtype=tf.float64)
        Bi = tf.get_variable("Bi", [M, ], dtype=tf.float64)

        p = tf.nn.embedding_lookup(P, u_in)
        q = tf.nn.embedding_lookup(Q, i_in)
        bu = tf.nn.embedding_lookup(Bu, u_in)
        bi = tf.nn.embedding_lookup(Bi, i_in)
        rhat = tf.reduce_sum(tf.multiply(p, q), axis=1) + bu + bi

        # training network
        rmse = tf.losses.mean_squared_error(r_in, rhat)
        reg = tf.nn.l2_loss(p) + tf.nn.l2_loss(q) + tf.nn.l2_loss(bu) + tf.nn.l2_loss(bi)
        loss = tf.cast(rmse, tf.float64) + tf.reduce_mean(lamb*reg)
        opt = tf.train.RMSPropOptimizer(lr, decay).minimize(loss)

        # run training
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        sess.run([user_iterator.initializer, item_iterator.initializer, target_iterator.initializer],
                      feed_dict={user_in: np.array(Users_train), item_in: np.array(Items_train), rate_in:np.array(Ratings_train)})
        num_training_steps = len(Users_train) * numepochs // batchsize

        for i in range(num_training_steps):
            _, e, c = sess.run([opt, rmse, loss])
            if (i % print_every == 0):
                print('Step %i: Minibatch Loss: %f, RMSE: %f' % (i, c, np.sqrt(e)))

        # testing
        user_iterator, u_in = makeTfDataset(user_in, len(Users_test), 1)
        item_iterator, i_in = makeTfDataset(item_in, len(Items_test), 1)
        target_iterator, r_in = makeTfDataset(rate_in, len(Ratings_test), 1)
        sess.run([user_iterator.initializer, item_iterator.initializer, target_iterator.initializer],
                 feed_dict={user_in: np.array(Users_test), item_in: np.array(Items_test),
                            rate_in: np.array(Ratings_test)})
        e = sess.run([rmse])
        print('Test: RMSE: %f ' % (np.sqrt(e)))
        sess.close()
