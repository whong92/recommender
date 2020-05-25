import numpy as np
import os
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Activation, Add, Subtract
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
import tensorflow as tf
from .PMF import AdaptiveRegularizer, RegularizerInspector, PMFLoss, ReducedEmbedding
import parse

from datetime import datetime
from ..utils.ItemMetadata import ExplicitDataFromCSV
import json
from pprint import pprint
import jinja2

def parse_config_json(config_json, data):
    
    with open(config_json, 'r') as fp: s = fp.read()
    config_s = jinja2.Template(s).render(data=data)
    print(config_s)
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
    return config[0]

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
    model_conf = None

    for layer_conf in config:

        layer_type = layer_conf['type']
        layer_name = layer_conf['name']
        layer_params = layer_conf['params']
        layer_inputs = layer_conf.get('inputs', [])
        layer = None
        L = None
        
        if layer_type == 'model': # differ until the end
            model_conf = layer_conf
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
            raise Exception("Layer type not recognized!")

        if layer is None: # call layer
            inputs = [name2layers[name]['output'] for name in layer_inputs]
            y = L(*inputs)
            layer = {'layer': L, 'output': y}
        
        layers.append(layer)
        name2layers[layer_name] = layer
        continue

    # default training params
    epochs = model_conf.get('epochs', 30)
    batchsize = model_conf.get('batchsize', 500)
    # TODO: this should really be a property of the data object??
    normalize = model_conf.get('normalize', None)
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

    return model, model_conf

class MatrixFactorizer(object):

    MODEL_FORMAT = 'model.{epoch:03d}-{val_loss:.2f}.h5'

    def __init__(
        self, model_dir, N, M, Nranked, mode='train', config_path=None, saved_model=None,
    ):
        data_conf = {'N': N, 'M': M, 'Nranked': Nranked}
        self.model_path = model_dir
        self.model, self.model_conf, self.start_epoch = MatrixFactorizer.intialize_from_json(model_dir, data_conf, saved_model=saved_model,  config_path=config_path)

    @staticmethod
    def intialize_from_json(model_path, data_conf, saved_model=None,  config_path=None,):

        start_epoch = 0

        assert config_path is not None, "config not provided for model"
        config = parse_config_json(config_path, data=data_conf)
        model_conf = get_model_config(config)

        # check for explicitly saved model
        if saved_model is not None: 
            model = tf.keras.models.load_model(saved_model, compile=True)
            model.summary(line_length=88)
            return model, model_conf, start_epoch
        
        # look for checkpoints in model_path
        ckpts = map(lambda x: parse.parse(MatrixFactorizer.MODEL_FORMAT, x), os.listdir(model_path))
        ckpts = list(sorted(filter(lambda x: x is not None, ckpts), key=lambda x: x['epoch'], reverse=True))
        if len(ckpts) > 0:
            ckpt = ckpts[0]
            start_epoch = ckpt['epoch']
            model_name = os.path.join(model_path, MatrixFactorizer.MODEL_FORMAT.format(epoch=ckpt['epoch'], val_loss=ckpt['val_loss']))
            model = tf.keras.models.load_model(model_name, compile=True)
            model.summary(line_length=88)
            return model, model_conf, start_epoch
        
        # otherwise make model from scratch
        print("Initializing model from scratch")
        model, model_conf = make_model(config)

        return model, model_conf, start_epoch

    def fit(
        self, 
        data: ExplicitDataFromCSV,
        early_stopping=True, tensorboard=True,
        extra_callbacks=None
    ):

        (u_train, i_train, r_train), (u_test, i_test, r_test) = data.make_training_datasets(dtype='dense')

        batchsize = self.model_conf['batchsize']
        epochs = self.model_conf['epochs']
        normalize = self.model_conf['normalize']
        
        u_all = np.concatenate([u_train, u_test])
        i_all = np.concatenate([i_train, i_test])
        r_all = np.concatenate([r_train, r_test])
        if normalize: r_all = (r_all - normalize['loc'])/normalize['scale']
        val_split = float(len(u_test))/len(u_all)

        if tensorboard:
            tensorboard_path = os.path.join(self.model_path, 'tensorboard_logs')
            file_writer = tf.summary.create_file_writer(tensorboard_path + "/metrics")
            file_writer.set_as_default()
        
        self.model.fit(
            {'u_in': u_all, 'i_in': i_all}, {'rhat': r_all},
            epochs=epochs, batch_size=batchsize, verbose=1, initial_epoch=self.start_epoch,
            shuffle=True, validation_split=val_split,
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
        return self.model.predict({'u_in': u, 'i_in': i}, batch_size=5000)

    def add_users(self, num=1):
        raise NotImplementedError

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