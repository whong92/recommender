import tensorflow as tf
import numpy as np
import os

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
                P = tf.get_variable("P", [N, f], dtype=tf.float64, initializer=tf.random_normal_initializer)
                Q = tf.get_variable("Q", [M, f], dtype=tf.float64, initializer=tf.random_normal_initializer)

                p = tf.nn.embedding_lookup(P, u_in)
                q = tf.nn.embedding_lookup(Q, i_in)
                dotted = tf.keras.layers.Dot(1)([p,q])
                y = tf.squeeze(dotted)
                rhat = y

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {'rhat': rhat, 'p':p, 'q':q}
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            r_in = tf.cast(features['rating'], tf.float64)

            lamb = params['lamb']
            lr = params['lr']
            decay = params['decay']

            mse = tf.losses.mean_squared_error(r_in, rhat)
            reg = tf.nn.l2_loss(p) + tf.nn.l2_loss(q)
            loss = tf.cast(mse, tf.float64) + tf.reduce_mean(lamb * reg)
            logging_hook = tf.train.LoggingTensorHook({"mse": mse,
                                                       "loss": loss, "reg":reg}, every_n_iter=100)

            if mode == tf.estimator.ModeKeys.EVAL:
                metrics = {'rmse': tf.metrics.mean_squared_error(r_in, rhat)}
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)

            assert mode == tf.estimator.ModeKeys.TRAIN
            opt = tf.train.RMSPropOptimizer(lr, decay).minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=opt, training_hooks=[logging_hook])

    @staticmethod
    def _np_input_fn(features, batchsize=5000, numepochs=30):
        features = dict(features)
        dataset = tf.data.Dataset.from_tensor_slices(features)
        dataset = dataset.repeat(numepochs).batch(batchsize)
        return dataset

    @staticmethod
    def _tfr_input_fn(filenames, batchsize=5000, numepochs=30):
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

    def initialize(self, model_dir, N, M, f=10, lr=0.01, lamb=0.01, decay=0.0):
        self.model = tf.estimator.Estimator(
            model_fn=MatrixFactorizer._model_fn,
            params={
                'N': N, 'M': M, 'f': f, 'lr': lr, 'lamb': lamb, 'decay': decay
            }, model_dir=os.path.join(model_dir, 'tmp'))
        return

    def __init__(self, model_dir, N, M, f=10, lr=0.01, lamb=0.01, decay=0.0):
        self.initialize(model_dir, N, M, f, lr, lamb, decay)

    def fit(self, ds_train, ds_test, batchsize=5000, numepochs=10):
        # TODO: holy crap this is fucking ugly
        self.model.train(input_fn=lambda:ds_train().repeat(numepochs).batch(batchsize))
        #Evaluate the model
        eval_result = self.model.evaluate(input_fn=lambda: ds_test().batch(batchsize))
        print('Test: RMSE: %f ' % np.sqrt(eval_result['rmse']))

    # TODO: add predict mode
    def save(self, path):
        return self.model.export_saved_model(export_dir_base=path, serving_input_receiver_fn=MatrixFactorizer._predict_input_fn)


class MatrixFactorizerBias(MatrixFactorizer):

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
                P = tf.get_variable("P", [N, f], dtype=tf.float64, initializer=tf.random_normal_initializer)
                Q = tf.get_variable("Q", [M, f], dtype=tf.float64, initializer=tf.random_normal_initializer)
                Bu = tf.get_variable("Bu", [N, ], dtype=tf.float64, initializer=tf.random_normal_initializer)
                Bi = tf.get_variable("Bi", [M, ], dtype=tf.float64, initializer=tf.random_normal_initializer)

                p = tf.nn.embedding_lookup(P, u_in)
                q = tf.nn.embedding_lookup(Q, i_in)
                bu = tf.nn.embedding_lookup(Bu, u_in)
                bi = tf.nn.embedding_lookup(Bi, i_in)
                dotted = tf.keras.layers.Dot(1)([p, q])
                y = tf.squeeze(dotted)
                rhat = y + bu + bi

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {'rhat': rhat, 'p': p, 'q': q}
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

    def initialize(self, model_dir, N, M, f=10, lr=0.01, lamb=0.01, decay=0.0):
        self.model = tf.estimator.Estimator(
            model_fn=MatrixFactorizerBias._model_fn,
            params={
                'N': N, 'M': M, 'f': f, 'lr': lr, 'lamb': lamb, 'decay': decay
            }, model_dir=os.path.join(model_dir, 'tmp'))
        return

