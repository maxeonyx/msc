from typing import Callable, Collection, Tuple, TypedDict, Union, List
from math import pi, tau

import tensorflow as tf
import einops as ein

import typing
if typing.TYPE_CHECKING:
    import keras.api._v2.keras as keras
    from keras.api._v2.keras import Model, Input, layers
else:
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers

from ._layer_utils import input_dict, make_causal_mask, shape_list

def codebook(num_tokens, num_dims, seq_dims=[]) -> Model:

    embedder = layers.Embedding(num_tokens, num_dims, name="codebook")

    def call(inputs):
        return embedder(inputs["token"])

    inputs = input_dict(
        Input(shape=[*seq_dims], dtype=tf.int32, name="token"),
    )
    
    return Model(inputs=inputs, outputs=call(inputs), name="embd")

def positional_embedding(seq_len, embd_dim) -> Model:
    assert embd_dim % 2 == 0, f"embd_dim must be divisible by 2 to use positional encoding, got embd_dim={embd_dim}"
    
    i = tf.range(embd_dim//2)
    i = tf.cast(i, dtype=tf.float32)
    i = tf.expand_dims(i, -2)
    
    scale = tf.pow(scale, 2.*i/embd_dim) 
    
    def call(vals):
        vals = tf.expand_dims(vals, -1)
        vals = tf.cast(vals, tf.float32)
        # the bit inside the sin / cos
        rate = pi*vals / scale
        sin = tf.sin(rate)
        cos = tf.cos(rate)
        encoding = tf.concat([sin, cos], axis=-1)
        return encoding

    inputs = input_dict(
        Input(shape=[], dtype=tf.int32, name="index"),
    )
    
    return Model(inputs=inputs, outputs=call(inputs), name="embd")

def angle_embedding(embd_dim: int, num_repeats: int) -> Model:
    """
    Embed angles as unit vectors. Create num_repeats copies of them rotated
    evenly around one quarter of the unit circle.
    """

    assert embd_dim % 2 == 0, f"embd_dim must be divisible by 2 to use angle embedding, got embd_dim={embd_dim}"
    assert embd_dim % num_repeats == 0, f"embd_dim must be divisible by num_repeats to use angle embedding, got embd_dim={embd_dim} and num_repeats={num_repeats}"

    def angle_call(angles):
        scale = (tau / 4.) * (1. / num_repeats) # only need to produce rotations up to tau/4, because the model can easily invert angles
        offset = tf.range(num_repeats, dtype=tf.float32) * tf.constant([scale])

        offsets = tf.tile(offset, [*shape_list(angles), 1])
        angles = tf.tile(angles, [*tf.ones_like(tf.shape(angles)), num_repeats])
        angles = angles + offsets
        angles = tf.stack([tf.sin(angles), tf.cos(angles)], axis=-1)
        angles = ein.rearrange(angles, '... repeats sincos -> ... (repeats sincos)')

        return angles

    inputs = input_dict(
        Input(shape=[], dtype=tf.float32, name="angles"),
    )
    
    return Model(inputs=inputs, outputs=angle_call(inputs), name="embd")
