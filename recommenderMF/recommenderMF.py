import tensorflow as tf
from ..utils.utils import csv2df, df2umCSR, rmse, mean_nnz
import numpy as np

if __name__=="__main__":
    train_test_split = 0.8
    df, N, M = csv2df('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv', 'userId', 'movieId', 'rating')
    perm = np.random.permutation(len(df))
    D_train = df.iloc[perm[:int(len(df) * train_test_split)]]
    Users = D_train['user']
    Items = D_train['item']
    Ratings = D_train['rating']
    #U_train = df2umCSR(df.iloc[perm[:int(len(df) * train_test_split)]], M, N)
    #U_test = df2umCSR(df.iloc[perm[int(len(df) * train_test_split):]], M, N)

    print_every=50

    f = 10 # latent factor dimensionality
    lamb = 0.01
    batchsize = 32
    numepochs = 10
    lr = 0.01
    decay = 0.0
    mu = np.mean(np.array(Ratings))
    model_vars = {}

    with tf.variable_scope("MF", reuse=tf.AUTO_REUSE):

        P = tf.get_variable("P", [N, f], dtype=tf.float64)
        Q = tf.get_variable("Q", [M, f], dtype=tf.float64)
        Bu = tf.get_variable("Bu", [N, ], dtype=tf.float64)
        Bi = tf.get_variable("Bi", [M, ], dtype=tf.float64)

        # input definition
        user_in = tf.placeholder(tf.int32, [None,])
        item_in = tf.placeholder(tf.int32, [None,])
        rate_in = tf.placeholder(tf.int32, [None,])
        # user input
        user = tf.data.Dataset.from_tensor_slices(user_in)
        user = user.batch(batchsize)
        user = user.repeat(numepochs)
        user_iterator = user.make_initializable_iterator()
        u_in = user_iterator.get_next()
        # item input
        item = tf.data.Dataset.from_tensor_slices(item_in)
        item = item.batch(batchsize)
        item = item.repeat(numepochs)
        item_iterator = item.make_initializable_iterator()
        i_in = item_iterator.get_next()
        # target
        target = tf.data.Dataset.from_tensor_slices(rate_in)
        target = target.batch(batchsize)
        target = target.repeat(numepochs)
        target_iterator = target.make_initializable_iterator()
        r_in = target_iterator.get_next()

        # training network
        p = tf.nn.embedding_lookup(P, u_in)
        q = tf.nn.embedding_lookup(Q, i_in)

        bu = tf.nn.embedding_lookup(Bu, u_in)
        bi = tf.nn.embedding_lookup(Bi, i_in)
        rhat = tf.reduce_sum(tf.multiply(p, q), axis=1) + bu + bi
        rmse = tf.losses.mean_squared_error(r_in, rhat)
        reg = tf.nn.l2_loss(p) + tf.nn.l2_loss(q) + tf.nn.l2_loss(bu) + tf.nn.l2_loss(bi)
        loss = tf.cast(rmse, tf.float64) + tf.reduce_mean(lamb*reg)
        opt = tf.train.RMSPropOptimizer(lr, decay).minimize(loss)

        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        sess.run([user_iterator.initializer, item_iterator.initializer, target_iterator.initializer],
                      feed_dict={user_in: np.array(Users), item_in: np.array(Items), rate_in:np.array(Ratings)})
        num_training_steps = len(Users) * numepochs // batchsize

        for i in range(num_training_steps):
            _, c = sess.run([opt, loss])
            if (i % print_every == 0):
                print('Step %i: Minibatch Loss: %f' % (i, c))
