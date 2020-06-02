import numpy as np
import os
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Activation, Add, Subtract
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
import tensorflow as tf
from .PMF import AdaptiveRegularizer, RegularizerInspector, PMFLoss, ReducedEmbedding
from ..utils.utils import get_pos_ratings_padded, mean_nnz
import scipy.sparse as sps
import parse

from datetime import datetime
from ..utils.ItemMetadata import ExplicitDataFromCSV
import json
from pprint import pprint
from jinja2 import Environment, PackageLoader, select_autoescape
from tqdm import tqdm

def parse_config_json(config_json, data):
    
    env = Environment(
        loader=PackageLoader('reclibwh', 'core/model_templates'), autoescape=select_autoescape(['json'])
    )
    config_s = env.get_template(config_json).render(data=data)
    config = json.loads(config_s)
    assert type(config) is list, "config needs to be a list of dicts"
    
    for i, c in enumerate(config):
        assert type(c) is dict, "config needs to be a list of dicts"
        assert 'type' in c, "need to specify type for all layers"
        assert 'params' in c, "need to specify params for all layers"
        if not c['type']=="input":
            assert 'inputs' in c, "required to specify input names to all layers"
        if i==0: 
            assert c['type']=='model'
            assert 'outputs' in c, "required to specify output names to model"
        if 'name' not in c: c['name'] = 'layer_{:03d}'.format(i)
    
    return config

def make_regularizer(reg_conf=None):
    """[convenience function to make a 'standard' keras regularizer from some a regularizer config]

    Keyword Arguments:
        reg_conf {[dict]} -- [dictionary for the regularizer configuration] (default: {None})

    Raises:
        Exception: [if regularizer not recognized]

    Returns:
        [keras.regularizer] -- [a standard regularizer object]
    """

    if reg_conf is None: return None
    reg_type = reg_conf['type']
    params = reg_conf['params']
    if reg_type == "l2": return regularizers.l2(**params)
    elif reg_type == "l1": return regularizers.l2(**params)
    else: raise Exception("regularizer not recognized")

def get_model_config(config):
    return list(filter(lambda x: x['type']=='model',config))

def make_model(config: dict):
    """[
        function that takes in a model config as a dictionary and makes the 
        model. For now it instantiates an RMSProp model by default. Will add 
        options for other optimizers soon]

    Arguments:
        config {dict} -- [the dictionary describing the model configuration]

    Raises:
        Exception: [if shenanigans]

    Returns:
        [keras.Model] -- [a compiled model]
    """

    input_names = {}
    name2layers = {}
    layers = []
    model_confs = []
    models = []

    for layer_conf in config:

        layer_type = layer_conf['type']
        layer_name = layer_conf['name']
        layer_params = layer_conf['params']
        layer_inputs = layer_conf.get('inputs', [])
        layer = None
        L = None
        
        if layer_type == 'model': # differ until the end
            model_confs.append(layer_conf)
            continue
        
        elif layer_type == 'input':
            L = Input(**layer_params, name=layer_name)
            layer = {'layer': L, 'output': L} # inputs are their own outputs

        elif layer_type == 'embedding':
            L = Embedding(**layer_params, embeddings_regularizer=make_regularizer(layer_conf.get("regularizer_params", None)), name=layer_name)
        
        elif layer_type == 'adaptive_regularizer':
            L = AdaptiveRegularizer(**layer_params, name=layer_name)
        
        elif layer_type == 'reduced_embedding':
            L = ReducedEmbedding(**layer_params, name=layer_name)

        elif layer_type == 'combine_embeddings':
            inputs = [name2layers[name]['output'] for name in layer_inputs]
            L = Flatten(name=layer_name)
            y = L(Dot(2)(inputs))
            layer = {'layer': L, 'output': y}

        elif layer_type == 'combine_sum':
            inputs = [name2layers[name]['output'] for name in layer_inputs]
            y = Add(name=layer_name)(inputs)
            layer = {'layer': L, 'output': y}
        
        elif layer_type == 'combine_sub':
            inputs = [name2layers[name]['output'] for name in layer_inputs]
            y = Subtract(name=layer_name)(inputs)
            layer = {'layer': L, 'output': y}

        elif layer_type == 'activation':
            L = Activation(**layer_params, name=layer_name)
            
        else:
            raise Exception("Layer type not recognized {:s}!".format(layer_type))

        if layer is None: # call layer
            inputs = [name2layers[name]['output'] for name in layer_inputs]
            y = L(*inputs)
            layer = {'layer': L, 'output': y}
        
        layers.append(layer)
        name2layers[layer_name] = layer
        continue

    for model_conf in model_confs:

        # default training params
        epochs = model_conf.get('epochs', 30)
        batchsize = model_conf.get('batchsize', 500)
        # TODO: this should really be a property of the data object??
        lr = model_conf.get('lr', 0.01)
        
        model_inputs = [name2layers[name]['output'] for name in model_conf['inputs']]
        model_outputs = [name2layers[name]['output'] for name in model_conf['outputs']]
        pred_output = model_conf['outputs'][0]

        model = Model(inputs=model_inputs, outputs=model_outputs)
        model.compile(
            optimizer=optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.9),
            loss={pred_output: PMFLoss},
            metrics = {pred_output: 'mse'}
        )
        model.summary(line_length=88)
        models.append(model)

    return models, model_confs


class MatrixFactorizerDataIterator(object):

    def __init__(self, data: ExplicitDataFromCSV, batchsize: int, epochs: int, train: bool=False):
        self.data = data
        self.batchsize = batchsize
        self.epochs = epochs
        self.length = (self.data.Nranked//batchsize + 1)*epochs
        self.train = train
        self.bla = self.make_data()
    
    def make_data(self):
        train = self.train
        (u_train, i_train, r_train), (u_test, i_test, r_test) = self.data.make_training_datasets(dtype='dense')
        if train: users, items, ratings = u_train, i_train, r_train
        else:  users, items, ratings = u_test, i_test, r_test
        return (users, items, ratings)

    def iterate(self):
        
        batchsize = self.batchsize
        epochs = self.epochs
        users, items, ratings = self.bla

        num_iter = len(users)//batchsize + 1
        for epoch in range(epochs):
            for i in range(num_iter):
                s = i*batchsize
                e = min((i+1)*batchsize, len(users))
                u = users[s:e]
                i = items[s:e]
                r = ratings[s:e]
                yield {'u_in': u, 'i_in': i}, {'rhat': r}
        return
    
    def __len__(self):
        batchsize = self.batchsize
        epochs = self.epochs
        return int(np.ceil(len(self.bla[0])/batchsize)*epochs)


class AsymSVDDataIterator(MatrixFactorizerDataIterator):

    def make_data(self):
        train = self.train
        (u_train, i_train, r_train), (u_test, i_test, r_test) = self.data.make_training_datasets(dtype='dense')
        if train: users, items, ratings = u_train, i_train, r_train
        else:  users, items, ratings = u_test, i_test, r_test
        U, _ = self.data.make_training_datasets(dtype='sparse')
        return (users, items, ratings, U)
    
    def iterate(self):
        
        batchsize = self.batchsize
        epochs = self.epochs
        users, items, ratings, U = self.bla

        Ucsc = sps.csc_matrix(U)
        Bi = np.reshape(np.array(mean_nnz(Ucsc, axis=0, mu=0)), newshape=(-1,))
        num_iter = len(users)//batchsize + 1
        for epoch in range(epochs):
            for i in range(num_iter):
                s = i*batchsize
                e = min((i+1)*batchsize, len(users))
                u = users[s:e]
                i = items[s:e]
                r = ratings[s:e]
                rs, ys = get_pos_ratings_padded(U, u, 0, offset_yp=1)
                bs = Bi[ys-1]
                yield {'u_in': u, 'i_in': i, 'uj_in': ys, 'bj_in': bs, 'ruj_in': rs}, {'rhat': r}
        return


class LMFDataIterator(MatrixFactorizerDataIterator):

    def iterator(self, train=False):
        raise NotImplementedError
        

class MatrixFactorizer(object):

    MODEL_FORMAT = 'model.{epoch:03d}-{val_loss:.2f}.h5'

    def __init__(
        self, model_dir, N, M, Nranked, mode='train', config_path=None, saved_model=None,
    ):
        print(model_dir, config_path)
        data_conf = {'N': N, 'M': M, 'Nranked': Nranked}
        self.data_conf = data_conf
        self.model_path = model_dir
        self.config_path = config_path
        self.models, self.model_confs, self.start_epoch = MatrixFactorizer.intialize_from_json(model_dir, data_conf, saved_model=saved_model,  config_path=config_path)
        self.model = self.models[0]
        self.model_conf = self.model_confs[0]

    @staticmethod
    def intialize_from_json(model_path, data_conf, saved_model=None,  config_path=None):

        start_epoch = 0

        assert config_path is not None, "config not provided for model"
        config = parse_config_json(config_path, data=data_conf)
        model_confs = get_model_config(config)

        # check for explicitly saved model
        if saved_model is not None: 
            if type(saved_model) is not list: saved_model = [saved_model]
            models = []
            for s in saved_model:
                model = tf.keras.models.load_model(s, compile=True)
                model.summary(line_length=88)
                models.append(model)
            return models, model_confs, start_epoch
        
        # look for checkpoints in model_path
        ckpts = map(lambda x: parse.parse(MatrixFactorizer.MODEL_FORMAT, x), os.listdir(model_path))
        ckpts = list(sorted(filter(lambda x: x is not None, ckpts), key=lambda x: x['epoch'], reverse=True))
        if len(ckpts) > 0:
            ckpt = ckpts[0]
            start_epoch = ckpt['epoch']
            model_name = os.path.join(model_path, MatrixFactorizer.MODEL_FORMAT.format(epoch=ckpt['epoch'], val_loss=ckpt['val_loss']))
            model = tf.keras.models.load_model(model_name, compile=True)
            model.summary(line_length=88)
            return [model], [model_conf], start_epoch
        
        # otherwise make model from scratch
        print("Initializing model from scratch")
        models, model_confs = make_model(config)

        return models, model_confs, start_epoch
    
    def make_data_iterator(self, data: ExplicitDataFromCSV, train=True):
        batchsize = self.model_conf['batchsize'] if train else 5000
        epochs = self.model_conf['epochs']
        return MatrixFactorizerDataIterator(data, batchsize, epochs, train=train)

    def fit(
        self, 
        data: ExplicitDataFromCSV,
        early_stopping=True, tensorboard=True,
        extra_callbacks=None
    ):
        
        train_data_iter = self.make_data_iterator(data, train=True)
        val_data_iter = self.make_data_iterator(data, train=False)
        
        epochs = train_data_iter.epochs
        steps_per_epoch = len(train_data_iter)//epochs
        validation_batch_size = val_data_iter.batchsize
        validation_steps = len(val_data_iter)//val_data_iter.epochs

        if tensorboard:
            tensorboard_path = os.path.join(self.model_path, 'tensorboard_logs')
            file_writer = tf.summary.create_file_writer(tensorboard_path + "/metrics")
            file_writer.set_as_default()
        
        self.model.fit(
            train_data_iter.iterate(), 
            steps_per_epoch=steps_per_epoch, epochs=epochs,
            validation_data=val_data_iter.iterate(), 
            validation_batch_size=validation_batch_size, validation_steps=validation_steps, 
            verbose=1, initial_epoch=self.start_epoch,
            shuffle=True,
            callbacks=[
                ModelCheckpoint(os.path.join(self.model_path, MatrixFactorizer.MODEL_FORMAT)),
                CSVLogger(os.path.join(self.model_path, 'history.csv')),
            ] + 
            ([EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)] if early_stopping else []) +
            ([TensorBoard(log_dir=tensorboard_path)] if tensorboard else []) +
            (extra_callbacks if extra_callbacks is not None else [])
        )

    def save(self, model_path='model.h5'):
        self.model.save(os.path.join(self.model_path, model_path))

    def predict(self, u, i):
        # TODO: use model_conf here to check against input dict
        return self.model.predict({'u_in': u, 'i_in': i}, batch_size=5000)

    def add_users(self, num=1):
        raise NotImplementedError

    def evaluate(self, data: ExplicitDataFromCSV):

        tmp = self.model_conf['epochs']
        self.model_conf['epochs'] = 1
        validation_data = self.make_data_iterator(data, train=False)
        self.model_conf['epochs'] = tmp
        
        self.model.evaluate(validation_data.iterate())


def transplant_weights(model_from: Model, model_to: Model):
    
    assert len(model_from.layers)==len(model_to.layers)

    for layer_from, layer_to in zip(model_from.layers, model_to.layers):
    
        print("transplanting layer {:s} -> {:s} ".format(layer_from.name, layer_to.name))

        weights_from = layer_from.get_weights()
        weights_to = layer_to.get_weights()
        assert len(weights_from) == len(weights_to)
        for wf, wt in zip(weights_from, weights_to):
            assert len(wf.shape) == len(wt.shape)
            print("    transplanting weights {} -> {} ".format(wf.shape, wt.shape))
            common_hypercube = tuple([slice(0,min(sf, st)) for sf, st in zip(wf.shape, wt.shape)])
            wt[tuple(common_hypercube)] = wf[tuple(common_hypercube)]
        layer_to.set_weights(weights_to)

if __name__=="__main__":

    data_folder = '/home/ong/personal/recommender/data/ml-latest-small-2'
    model_folder = '/home/ong/personal/recommender/models/MF_tmp'
    save_path = os.path.join(model_folder, "MF_{:s}".format(datetime.now().strftime("%Y-%m-%d.%H-%M-%S")))

    model = MatrixFactorizer(
        model_dir=model_folder, N=d.N, M=d.M,
        config_path='/home/ong/personal/recommender/reclibwh/core/model_templates/SVD_reg.json.template'
    )

    model.fit(
        d
        # extra_callbacks=[RegularizerInspector(['lambda_p', 'lambda_q']), TensorBoard(log_dir)] # for demo purposes
    )