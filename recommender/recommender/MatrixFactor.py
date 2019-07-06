import tensorflow as tf
from ..utils.utils import csv2df, makeTfDataset, splitDf
import numpy as np

class MatrixFactorizer(object):

    @staticmethod
    def _model_fn(features, labels, mode, params):

        with tf.device('/cpu:0'):
            u_in = tf.cast(features['user'], tf.int32)
            i_in = tf.cast(features['item'], tf.int32)

            N = params['N']
            M = params['M']
            f = params['f']

            with tf.variable_scope("MF/model", reuse=tf.AUTO_REUSE):

                # model
                P = tf.get_variable("P", [N, f], dtype=tf.float64)
                Q = tf.get_variable("Q", [M, f], dtype=tf.float64)
                Bu = tf.get_variable("Bu", [N, ], dtype=tf.float64)
                Bi = tf.get_variable("Bi", [M, ], dtype=tf.float64)

                p = tf.nn.embedding_lookup(P, u_in)
                q = tf.nn.embedding_lookup(Q, i_in)
                bu = tf.nn.embedding_lookup(Bu, u_in)
                bi = tf.nn.embedding_lookup(Bi, i_in)
                y = tf.reduce_sum(tf.multiply(p, q), axis=1)
                rhat = y + bu + bi

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {'rhat': rhat, 'p':p, 'q':q}
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            r_in = tf.cast(features['rating'], tf.float64)

            lamb = params['lamb']
            lr = params['lr']
            decay = params['decay']

            mse = tf.losses.mean_squared_error(r_in, rhat)
            reg = tf.nn.l2_loss(p) + tf.nn.l2_loss(q) + tf.nn.l2_loss(bu) + tf.nn.l2_loss(bi)
            loss = tf.cast(mse, tf.float64) + tf.reduce_mean(lamb * reg)

            if mode == tf.estimator.ModeKeys.EVAL:
                metrics = {'rmse': tf.metrics.mean_squared_error(r_in, rhat)}
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)

            assert mode == tf.estimator.ModeKeys.TRAIN
            opt = tf.train.RMSPropOptimizer(lr, decay).minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=opt)

    @staticmethod
    def _np_input_fn(features, batchsize=32, numepochs=30):
        features = dict(features)
        dataset = tf.data.Dataset.from_tensor_slices(features)
        dataset = dataset.repeat(numepochs).batch(batchsize)
        return dataset

    @staticmethod
    def _tfr_input_fn(filenames, batchsize=32, numepochs=30):
        raw_dataset = tf.data.TFRecordDataset(filenames)
        feature_description = {
            'user': tf.FixedLenFeature([], tf.int64),
            'item': tf.FixedLenFeature([], tf.int64),
            'rating': tf.FixedLenFeature([], tf.float32),
        }
        #raw_dataset = raw_dataset.shuffle(10000)
        raw_dataset = raw_dataset.repeat(numepochs).batch(batchsize)
        parsed_dataset = raw_dataset.map(lambda x: tf.parse_example(x, feature_description))
        return parsed_dataset

    @staticmethod
    def _predict_input_fn():
        input = {
            'user': tf.placeholder(dtype=tf.int32, shape=(None,)),
            'item': tf.placeholder(dtype=tf.int32, shape=(None,)),
        }
        return tf.estimator.export.ServingInputReceiver(input, input)

    def __init__(self, N, M, f=10, lr=0.01, lamb=0.01, decay=0.0):

        self.model = tf.estimator.Estimator(
            model_fn=MatrixFactorizer._model_fn,
            params={
                'N': N, 'M':M, 'f':f, 'lr':lr, 'lamb':lamb, 'decay':decay
            })
        return


    def fit_array_input(self, features_train, feature_test, batchsize=32, numepochs=10):
        self.model.train(
            input_fn=lambda:MatrixFactorizer._np_input_fn(features_train, batchsize, numepochs),
        )
        # Evaluate the model.
        eval_result = self.model.evaluate(
            input_fn=lambda: MatrixFactorizer._np_input_fn(feature_test, batchsize, 1),
        )
        print('Test: RMSE: %f ' % np.sqrt(eval_result['rmse']))


    def fit_tfr_input(self, train_filenames, test_filenames, batchsize=32, numepochs=10):
        self.model.train(
            input_fn=lambda: MatrixFactorizer._tfr_input_fn(train_filenames, batchsize, numepochs),
        )

        # Evaluate the model.
        eval_result = self.model.evaluate(
            input_fn=lambda: MatrixFactorizer._tfr_input_fn(test_filenames, 1000, 1),
        )
        print('Test: RMSE: %f ' % np.sqrt(eval_result['rmse']))

    # TODO: add predict mode
    def save(self, path):
        return self.model.export_saved_model(export_dir_base=path, serving_input_receiver_fn=MatrixFactorizer._predict_input_fn)


if __name__=="__main__":

    # hyperparamaters
    train_test_split = 0.8
    f = 10  # latent factor dimensionality
    lamb = 0.01
    batchsize = 32
    numepochs = 10
    lr = 0.01
    decay = 0.0

    # load data
    df, N, M = csv2df('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv', 'movieId', 'userId', 'rating')
    D_train, D_test = splitDf(df, train_test_split)
    print(N, M)
    Users_train = D_train['user']
    Items_train = D_train['item']
    Ratings_train = D_train['rating']

    print(len(D_train['rating']))

    Users_test = D_test['user']
    Items_test = D_test['item']
    Ratings_test = D_test['rating']

    """
    import pandas as pd
    user_map_df = pd.DataFrame.from_csv('D:/PycharmProjects/recommender/data/tmp/user_map_df.csv')
    item_map_df = pd.DataFrame.from_csv('D:/PycharmProjects/recommender/data/tmp/item_map_df.csv')
    N = len(user_map_df)
    M = len(item_map_df)
    """

    mf = MatrixFactorizer(N, M, f, lr, lamb, decay)
    tf.logging.set_verbosity(tf.logging.INFO)

    mf.fit_array_input(
        {'user':np.array(D_train['user'], dtype=np.int32), 'item':np.array(D_train['item'], dtype=np.int32), 'rating':np.array(D_train['rating'], dtype=np.float64)},
        {'user': np.array(D_test['user'], dtype=np.int32), 'item':np.array(D_test['item'], dtype=np.int32), 'rating':np.array(D_test['rating'], dtype=np.float64)},
    )

    """
    mf.fit_tfr_input(
        ['D:/PycharmProjects/recommender/data/tmp/bla_train{:03d}.tfrecord'.format(i) for i in range(1)],
        ['D:/PycharmProjects/recommender/data/tmp/bla_test{:03d}.tfrecord'.format(i) for i in range(1)]
    )
    """

    mf.save("./")

    """
    from tensorflow.contrib import predictor

    predict_fn = predictor.from_saved_model("./1554590618")
    print(predict_fn({
        'u_in': np.array([0 ,32, 55]),
        'i_in': np.array([0, 0, 0])
    }))
    """
