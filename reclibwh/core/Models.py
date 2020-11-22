import json
import os

from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Activation, Add, Subtract
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers, optimizers
import tensorflow as tf
from .PMF import AdaptiveRegularizer, RegularizerInspector, PMFLoss, ReducedEmbedding
import parse
from .Losses import PLMFLoss

from jinja2 import PackageLoader, select_autoescape
from jinja2 import Environment as JJEnvironment
from typing import Optional, List

STANDARD_KERAS_SAVE_FMT = 'model.{epoch:03d}-{val_loss:.2f}.h5'

def parse_config_json(config_json, data):

    env = JJEnvironment(
        loader=PackageLoader('reclibwh', 'core/model_templates'), autoescape=select_autoescape(['json'])
    )
    config_s = env.get_template(config_json).render(data=data)
    config = json.loads(config_s)
    assert type(config) is list, "config needs to be a list of dicts"

    for i, c in enumerate(config):
        assert type(c) is dict, "config needs to be a list of dicts"
        assert 'type' in c, "need to specify type for all layers"
        assert 'params' in c, "need to specify params for all layers"
        if not c['type' ]=="input":
            assert 'inputs' in c, "required to specify input names to all layers"
        if i== 0:
            assert c['type'] == 'model'
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
    if reg_type == "l2":
        return regularizers.l2(**params)
    elif reg_type == "l1":
        return regularizers.l2(**params)
    else:
        raise Exception("regularizer not recognized")


def get_model_config(config):
    return list(filter(lambda x: x['type'] == 'model', config))


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

        if layer_type == 'model':  # differ until the end
            model_confs.append(layer_conf)
            continue

        elif layer_type == 'input':
            L = Input(**layer_params, name=layer_name)
            layer = {'layer': L, 'output': L}  # inputs are their own outputs

        elif layer_type == 'embedding':
            L = Embedding(**layer_params,
                          embeddings_regularizer=make_regularizer(layer_conf.get("regularizer_params", None)),
                          name=layer_name)

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

        if layer is None:  # call layer
            inputs = [name2layers[name]['output'] for name in layer_inputs]
            y = L(*inputs)
            layer = {'layer': L, 'output': y}

        layers.append(layer)
        name2layers[layer_name] = layer
        continue

    for model_conf in model_confs:
        # default training params
        lr = model_conf.get('lr', 0.01)

        model_inputs = [name2layers[name]['output'] for name in model_conf['inputs']]
        model_outputs = [name2layers[name]['output'] for name in model_conf['outputs']]
        pred_output = model_conf['outputs'][0]

        loss = {
            'PMFLoss': PMFLoss,
            'PLMFLoss': PLMFLoss,
        }[model_conf['loss']]

        model = Model(inputs=model_inputs, outputs=model_outputs)
        model.compile(
            optimizer=optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.9),
            loss={pred_output: loss},
            metrics={pred_output: 'mse'}
        )
        model.summary(line_length=88)
        models.append(model)

    return models, model_confs


def model_restore(environment_path=None, saved_model=None, save_fmt=None) -> Optional[dict]:

    start_epoch = 0
    # check for explicitly saved model
    if saved_model is not None:
        if type(saved_model) is not list: saved_model = [saved_model]
        models = []
        for s in saved_model:
            print("restoring model from saved model {:s}".format(s))
            model = tf.keras.models.load_model(s, compile=True)
            model.summary(line_length=88)
            models.append(model)
        return {'model': models, 'start_epoch': start_epoch}

    # TODO: figure out how to restore multiple checkpoints, maybe pass in multiple save_fmts?
    # look for checkpoints in model_path
    ckpts = map(lambda x: parse.parse(save_fmt, x), os.listdir(environment_path))
    ckpts = list(sorted(filter(lambda x: x is not None, ckpts), key=lambda x: x['epoch'], reverse=True))
    if len(ckpts) > 0:
        ckpt = ckpts[0]
        start_epoch = ckpt['epoch']
        model_name = os.path.join(environment_path, save_fmt.format(epoch=ckpt['epoch'], val_loss=ckpt['val_loss']))
        model = tf.keras.models.load_model(model_name, compile=True)
        print("restoring checkpoint from {:s}".format(model_name))
        model.summary(line_length=88)
        return {'model': [model], 'start_epoch': start_epoch}

    print("model_restore: no checkpoints found")
    return None


def initialize_from_json(data_conf=None, config_path=None, config_override=None):

    # otherwise make model from scratch
    print("Initializing model from scratch")
    if not config_override: config_override = {}
    config = parse_config_json(config_path, data_conf)
    for k, v in config_override.items():
        for c in config:
            if c['name'] == k: c.update(v)
    models, model_confs = make_model(config)
    return models