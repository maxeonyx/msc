from typing import Callable, Collection, Tuple, TypedDict, Union, List

import tensorflow as tf

import typing
if typing.TYPE_CHECKING:
    import keras.api._v2.keras as keras
    from keras.api._v2.keras import Model, Input, layers
else:
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers

from ._layer_utils import input_dict, make_causal_mask, shape_list
    
def residual(embd_dim, n_layers: int=None, make_layer: Callable[[int], tf.Module]=None, layers: list[tf.Module]=None, seq_dims=[], normalization='scale', dropout=0.1, name="residual") -> Model:

    making_layers = n_layers is not None and make_layer is not None
    using_layers = layers is not None
    
    assert not (making_layers and using_layers), "Cannot specify both `layers` and `n_layers` + `make_layer`"
    assert making_layers or using_layers, "Must specify either `layers` or `n_layers` + `make_layer`"

    if making_layers:
        layers = [make_layer(i) for i in range(n_layers)]

    def call(inputs):
        embd = inputs["embd"]
        for layer in layers:
            embd += layer(embd)
        return embd

    inputs = input_dict(
        Input(shape=[*seq_dims, embd_dim], name="embd")
    )

    return Model(inputs=inputs, outputs=call(inputs), name=name)
