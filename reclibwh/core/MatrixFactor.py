import numpy as np
import os
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
import tensorflow as tf
from .PMF import AdaptiveRegularizer, RegularizerInspector
import parse

from datetime import datetime
from ..utils.ItemMetadata import ExplicitDataFromCSV

class MatrixFactorizer(object):

    MODEL_FORMAT = 'model.{epoch:03d}-{val_loss:.2f}.h5'

    def __init__(
        self, model_dir, N, M, Nranked, 
        f=10, lr=0.01, lamb=0.01, decay=0.9, epochs=30, batchsize=5000, 
        bias=False, normalize=None, adaptive_reg=False, mode='train'
    ):
        self.initialize(model_dir, N, M, Nranked, f, lr, lamb, decay, bias=bias, epochs=epochs, batchsize=batchsize, normalize=normalize, adaptive_reg=adaptive_reg)
    
    @staticmethod
    @tf.function
    def PMFLoss(r_ui, rhat):
        return tf.reduce_sum(tf.square(r_ui-rhat))

    def initialize(
        self, model_path, N, M, Nranked,
        f=10, lr=0.01, lamb=0.01, decay=0.9, epochs=30, batchsize=5000, 
        bias=False, normalize=None, adaptive_reg=False,
        saved_model=None
    ):

        self.saved_model = saved_model
        self.model_path = model_path
        self.epochs = epochs
        self.batchsize = batchsize
        self.normalize = normalize
        self.start_epoch = 0

        if saved_model is not None: 
            self.model = tf.keras.models.load_model(self.saved_model, compile=True)
            self.model.summary(line_length=88)
            return
        
        # look for checkpoints in model_path
        
        ckpts = map(lambda x: parse.parse(MatrixFactorizer.MODEL_FORMAT, x), os.listdir(model_path))
        ckpts = list(sorted(filter(lambda x: x is not None, ckpts), key=lambda x: x['epoch'], reverse=True))
        if len(ckpts) > 0:
            ckpt = ckpts[0]
            self.start_epoch = ckpt['epoch']
            model_name = os.path.join(self.model_path, MatrixFactorizer.MODEL_FORMAT.format(epoch=ckpt['epoch'], val_loss=ckpt['val_loss']))
            self.model = tf.keras.models.load_model(
                model_name, custom_objects={'AdaptiveRegularizer': AdaptiveRegularizer, 'PMFLoss': MatrixFactorizer.PMFLoss}, compile=True)
            self.model.summary(line_length=88)
            return

        u_in = Input(shape=(1,), dtype='int32', name='u_in')
        i_in = Input(shape=(1,), dtype='int32', name='i_in')

        P = Embedding(N, f, dtype='float32',
                      embeddings_regularizer=regularizers.l2(lamb) if not adaptive_reg and lamb>0. else None, 
                      input_length=1,
                      embeddings_initializer='random_normal', name='P')
        Q = Embedding(M, f, dtype='float32',
                      embeddings_regularizer=regularizers.l2(lamb) if not adaptive_reg and lamb>0. else None, 
                      input_length=1,
                      embeddings_initializer='random_normal', name='Q')

        p = P(u_in)
        q = Q(i_in)

        activation = 'linear' if normalize is None else 'sigmoid'
        if bias:
            Bu = Embedding(N, 1, dtype='float32', embeddings_initializer='random_normal')
            Bi = Embedding(M, 1, dtype='float32', embeddings_initializer='random_normal')
            bp = Bu(u_in)
            bq = Bi(i_in)
            rhat = Activation(activation, name='rhat')(Flatten()(Dot(2)([p, q])) + bp + bq)
        else:
            rhat = Activation(activation, name='rhat')(Flatten()(Dot(2)([p, q])))
        
        outputs=[rhat, p, q]
        if adaptive_reg:
            LambdaP = AdaptiveRegularizer(N, f, Nranked, initial_value=-15., alpha=0.01, name='lambda_p')
            LambdaQ = AdaptiveRegularizer(M, f, Nranked, initial_value=-15., alpha=0.01, name='lambda_q')
            rp2 = LambdaP(p)
            rq2 = LambdaQ(q)
            outputs += [rp2, rq2]

        self.model = Model(inputs=[u_in, i_in], outputs=outputs)
        self.model.compile(
            # optimizer=optimizers.Adam(learning_rate=lr),#, rho=decay),
            optimizer=optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.9),
            loss={'rhat': MatrixFactorizer.PMFLoss},
            metrics = {'rhat': 'mse'}
        )
        self.model.summary(line_length=88)

        return

    def fit(
        self, 
        u_train, i_train, r_train, 
        u_test, i_test, r_test, 
        early_stopping=True, tensorboard=True,
        extra_callbacks=None
    ):
        # TODO: checkpointing!
        
        u_all = np.concatenate([u_train, u_test])
        i_all = np.concatenate([i_train, i_test])
        r_all = np.concatenate([r_train, r_test])
        if self.normalize: r_all = (r_all - self.normalize['loc'])/self.normalize['scale']
        val_split = float(len(u_test))/len(u_all)

        if tensorboard:
            tensorboard_path = os.path.join(self.model_path, 'tensorboard_logs')
            file_writer = tf.summary.create_file_writer(tensorboard_path + "/metrics")
            file_writer.set_as_default()

        self.model.fit(
            {'u_in': u_all, 'i_in': i_all}, {'rhat': r_all},
            epochs=self.epochs, batch_size=self.batchsize, verbose=1, initial_epoch=self.start_epoch,
            shuffle=True, validation_split=val_split,
            callbacks=[
                ModelCheckpoint(
                    os.path.join(self.model_path, MatrixFactorizer.MODEL_FORMAT),
                ),
                CSVLogger(os.path.join(self.model_path, 'history.csv')),
            ] + 
            ([EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)] if early_stopping else []) +
            ([TensorBoard(log_dir=tensorboard_path)] if tensorboard else []) +
            (extra_callbacks if extra_callbacks is not None else []) # + [RegularizerInspector(['lambda_p', 'lambda_q'])]
        )

        #Evaluate the model
        eval_result = self.model.evaluate(
            {'u_in': u_test, 'i_in': i_test}, {'rhat': r_test}, batch_size=10000
        )
        print('Test: RMSE: %f ' % np.sqrt(eval_result[0]))

    def save(self):
        self.model.save(os.path.join(self.model_path, 'model.h5'))

    def predict(self, u, i):
        return self.model.predict({'u_in': u, 'i_in': i})


if __name__=="__main__":

    data_folder = '/home/ong/personal/recommender/data/ml-latest-small-2'
    model_folder = '/home/ong/personal/recommender/models'
    save_path = os.path.join(model_folder, "MF_{:s}".format(datetime.now().strftime("%Y-%m-%d.%H-%M-%S")))
    log_dir = '/home/ong/personal/recommender/tmp/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    d = ExplicitDataFromCSV(True, data_folder=data_folder)
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()

    '/home/ong/personal/recommender/models'

    data_train, data_test = d.make_training_datasets(dtype='dense')
    u_train, i_train, r_train = data_train
    u_test, i_test, r_test = data_test

    # model = make_model(d.N, d.M, f=20, lr=0.005, Nranked=len(r_train))

    model = MatrixFactorizer(
        model_dir=model_folder, N=d.N, M=d.M, f=20, lr=0.005, lamb=0., Nranked=len(r_train), bias=False, batchsize=50, epochs=60, 
        normalize={'loc':0., 'scale':5.0}, adaptive_reg=True, 
    )

    model.fit(
        u_train, i_train, r_train,
        u_test, i_test, r_test,
        extra_callbacks=[RegularizerInspector(['lambda_p', 'lambda_q']), TensorBoard(log_dir)]
    )