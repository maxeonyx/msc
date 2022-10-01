from typing import Callable, Collection, Literal, Tuple, TypedDict, Union, List
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
from ._blocks import featurewise_dense
from mx.utils import Einshape

def codebook(n_tokens: int, embd_shape: Einshape, add_begin_token: bool = True, name="codebook") -> Model:

    embedder = layers.Embedding(n_tokens, embd_shape.f_product, name=f"{name}/embd")

    def call(tokens):

        embd = embedder(tokens)
        
        return embd

    inputs = input_dict(
        Input(shape=embd_shape.s_shape, dtype=tf.int32, name="tokens"),
    )
    
    return Model(inputs=inputs, outputs=call(**inputs), name=name)

def prepend_begin_token(input_shape: Einshape, axis: Union[Literal["first_sequence_dim"], int] = "first_sequence_dim", name="prepend_begin_token") -> Model:

    begin_token_embedding = layers.Embedding(1, input_shape.f_product, name=f"{name}/begin_embd")

    if axis == "first_sequence_dim":
        axis = input_shape.b_rank
    
    begin_token_shape = input_shape.with_sequence_dims({ "seq": 1 }).with_feature_dims({ "embd": input_shape.f_product })

    def call(embd):
        begin_token = begin_token_embedding(tf.zeros(begin_token_shape.b_s_shape, tf.int32))
        embd = tf.concat([begin_token, embd], axis=axis) # concat along first sequence dim
        return embd

    inputs = input_dict(
        Input(shape=input_shape.s_f_shape, dtype=tf.float32, name="embd"),
    )
    
    return Model(inputs=inputs, outputs=call(**inputs), name=name)

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

def angle_embedding(num_repeats: int, input_shape: Einshape, embd_shape: Einshape, name="angle_embd") -> Model:
    """
    Embed angles as unit vectors. Create many copies of them rotated
    evenly around one quarter of the unit circle.
    """

    assert embd_shape.f_product % 2 == 0, f"embd_dim must be divisible by 2 to use angle embedding, got embd_dim={embd_shape.f_product}"

    angles_shape = input_shape.append_feature_dim("repeats", num_repeats).append_feature_dim("sincos", 2)

    dense_out = featurewise_dense(in_dims=angles_shape, out_dims=embd_shape, name=f"{name}/out")

    def angle_call(angles):
        scale = (tau / 4.) * (1. / num_repeats) # only need to produce rotations up to tau/4, because the model can easily invert angles
        offsets = tf.range(num_repeats, dtype=tf.float32) * scale
        # add "repeats" dim
        angles = angles[..., None]
        angles = angles + tf.broadcast_to(offsets, tf.broadcast_dynamic_shape(tf.shape(offsets), tf.shape(angles)))
        # add "sincos" dim
        angles = tf.stack([tf.sin(angles), tf.cos(angles)], axis=-1)
        # flatten to "embd" dim
        embd = dense_out(angles)

        return embd

    inputs = input_dict(
        Input(shape=input_shape.s_f_shape, dtype=tf.float32, name="angles"),
    )
    
    return Model(inputs=inputs, outputs=angle_call(**inputs), name=name)
