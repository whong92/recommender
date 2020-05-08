import numpy as np
import os
from keras.layers import Input, Embedding, Dot, Flatten
from keras.models import Model, load_model
from keras import regularizers, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

class MatrixFactorizer(object):

    def __init__(self, model_dir, N, M, f=10, lr=0.01, lamb=0.01, decay=0.9, bias=False, epochs=30, batchsize=5000, mode='train'):
        self.mode = mode
        self.initialize(model_dir, N, M, f, lr, lamb, decay, bias=bias, epochs=epochs, batchsize=batchsize)

    def initialize(self, model_path, N, M, f=10, lr=0.01, lamb=0.01, decay=0.9, epochs=30, batchsize=5000, bias=False):

        self.model_path = model_path
        self.epochs = epochs
        self.batchsize = batchsize

        if self.mode == 'predict':
            self.model = load_model(self.model_path, compile=True)
            self.model.summary()
            return

        u_in = Input(shape=(1,), dtype='int32', name='u_in')
        i_in = Input(shape=(1,), dtype='int32', name='i_in')

        P = Embedding(N, f, dtype='float32',
                      embeddings_regularizer=regularizers.l2(lamb), input_length=1,
                      embeddings_initializer='random_normal', name='P')
        Q = Embedding(M, f, dtype='float32',
                      embeddings_regularizer=regularizers.l2(lamb), input_length=1,
                      embeddings_initializer='random_normal', name='Q')

        p = P(u_in)
        q = Q(i_in)

        if bias:
            Bu = Embedding(N, 1, dtype='float32', embeddings_initializer='random_normal')
            Bi = Embedding(M, 1, dtype='float32', embeddings_initializer='random_normal')
            bp = Bu(u_in)
            bq = Bi(i_in)
            rhat = Flatten(name='rhat')(Dot(2)([p, q])) + bp + bq
        else:
            rhat = Flatten(name='rhat')(Dot(2)([p, q]))

        self.model = Model(inputs=[u_in, i_in], outputs=[rhat, p, q])
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),#, rho=decay),
            loss={'rhat': 'mean_squared_error'},
            metrics = {'rhat': 'mse'}
        )
        self.model.summary(line_length=88)

        return

    def fit(self, u_train, i_train, r_train, u_test, i_test, r_test):
        u_all = np.concatenate([u_train, u_test])
        i_all = np.concatenate([i_train, i_test])
        r_all = np.concatenate([r_train, r_test])
        val_split = float(len(u_test))/len(u_all)
        self.model.fit(
            {'u_in': u_all, 'i_in': i_all}, {'rhat': r_all},
            epochs=self.epochs, batch_size=self.batchsize, verbose=1,
            shuffle=True, validation_split=val_split,
            callbacks=[
                EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
                ModelCheckpoint(
                    os.path.join(self.model_path, 'model.{epoch:03d}-{val_loss:.2f}.h5'),
                    monitor='val_loss', save_best_only=True,
                ),
                CSVLogger(os.path.join(self.model_path, 'history.csv')),
            ]
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
