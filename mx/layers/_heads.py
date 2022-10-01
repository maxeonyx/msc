import abc
from dataclasses import dataclass
from typing import Callable

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import typing

from ._layer_utils import input_dict
from ._blocks import featurewise_dense
from mx import layers
from mx.utils import Einshape

if typing.TYPE_CHECKING:
    import keras.api._v2.keras as keras
    from keras.api._v2.keras import Model, Input, layers
else:
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers

@dataclass
class PredictionHead(tf.Module):
    final_layer: tf.Module | None
    """
    The final learned layer, eg. logits for classification, or parameters for a distribution.
    For regression, this is None unless there is a special loss function.
    """

    loss_fn: tf.Module
    """
    The loss function. Takes two parameters `y_true` and:
    - regression: `output`
    - classification: `logits`
    - distribution: `params`
    """

    output_fns: dict[str, tf.Module] | None
    """
    Functions that transform the model output eg. sampling or distribution statistics.
    Typically None for regression, but can be used for classification or distribution outputs.
    """


def mse(in_dims: Einshape, name="mse") -> PredictionHead:

    out_dims = in_dims
    
    def loss_fn_call(y_true, output):
        return tf.reduce_mean(tf.square(y_true - output))
    
    loss_fn_inputs = input_dict(
        Input(shape=out_dims.s_f_shape, name="y_true"),
        Input(shape=out_dims.s_f_shape, name="output"),
    )
    loss_fn = Model(
        inputs=loss_fn_inputs,
        outputs=loss_fn_call(loss_fn_inputs),
        name="loss",
    )
    
    return PredictionHead(
        final_layer=None,
        loss_fn=loss_fn,
        output_fns=None,
        name=name,
    )


def circular_mse(embd_dims: Einshape, name="mse") -> PredictionHead:
    """
    Circular mean squared error. Takes y_true as angles in radians, and y_pred as unit vectors.
    """
    target_dims = embd_dims.with_feature_dims({})
    out_dims = embd_dims.with_feature_dims({ "sincos": 2 })

    final_layer_call = featurewise_dense(
        in_dims=embd_dims,
        out_dims=out_dims,
        name="final_layer",
    )
    final_layer_inputs = input_dict(
        Input(shape=embd_dims.s_f_shape, name="embd"),
    )
    final_layer = Model(
        inputs=final_layer_inputs,
        outputs=final_layer_call(final_layer_inputs),
        name="unit_vectors",
    )
    
    def loss_fn_call(y_true, unit_vectors):
        x = unit_vectors[..., 0]
        y = unit_vectors[..., 1]
        return tf.reduce_mean(tf.square(tf.math.sin(y_true) - y) + tf.square(tf.math.cos(y_true) - x))

    loss_fn_inputs = input_dict(
        Input(shape=target_dims.s_f_shape, name="y_true"),
        Input(shape=out_dims.s_f_shape, name="unit_vectors"),
    )
    loss_fn = Model(
        inputs=loss_fn_inputs,
        outputs=loss_fn_call(**loss_fn_inputs),
        name="loss",
    )

    unit_vectors = input_dict(
        Input(shape=out_dims.s_f_shape, name="unit_vectors"),
    )
    def angles_call(unit_vectors):
        x = unit_vectors[..., 0]
        y = unit_vectors[..., 1]
        return tf.math.atan2(y, x)
    
    output_fns = {
        "Angle": Model(
            inputs=unit_vectors,
            outputs=angles_call(**unit_vectors),
            name="angles",
        ),
    }
    
    return PredictionHead(
        final_layer=final_layer,
        loss_fn=loss_fn,
        output_fns=output_fns,
    )

def categorical(in_dims: Einshape, num_categories: int, name="categorical") -> PredictionHead:

    out_dims = in_dims.with_feature_dims({ "c": num_categories })

    final_layer_call = featurewise_dense(
        in_dims=in_dims,
        out_dims=out_dims,
        name="logits",
    )
    final_layer_inputs = input_dict(
        Input(shape=in_dims.s_f_shape, name="embd"),
    )
    final_layer = Model(
        inputs=final_layer_inputs,
        outputs=final_layer_call(final_layer_inputs),
        name="logits",
    )

    def dist(logits):
        return tfd.Categorical(logits=logits)

    def loss_fn_call(y_true, logits):
        d = dist(logits)
        return -tf.reduce_mean(d.log_prob(y_true))
    
    loss_fn_inputs = input_dict(
        Input(shape=out_dims.s_f_shape, name="y_true"),
        Input(shape=out_dims.s_f_shape, name="logits"),
    )
    loss_fn = Model(
        inputs=loss_fn_inputs,
        outputs=loss_fn_call(loss_fn_inputs),
        name="loss",
    )

    def mode_call(logits):
        return tf.argmax(logits, axis=-1)
    
    def sample_call(logits):
        return dist(logits).sample()
    
    def entropy_call(logits):
        return dist(logits).entropy()

    logits = input_dict(
        Input(shape=out_dims.s_f_shape, name="logits"),
    )
    
    output_fns = {
        "Mode": Model(
            inputs=logits,
            outputs=mode_call(logits),
            name="mode",
        ),
        "Sample": Model(
            inputs=logits,
            outputs=sample_call(logits),
            name="sample",
        ),
        "Entropy": Model(
            inputs=logits,
            outputs=entropy_call(logits),
            name="entropy",
        ),
    }

    return PredictionHead(
        final_layer=final_layer,
        loss_fn=loss_fn,
        output_fns=output_fns,
        name=name,
    )
