from enum import Enum

from mx.prelude import *
from ._blocks import featurewise_dense
from mx.utils import Einshape

@export
def codebook(n_tokens: int, embd_shape: Einshape, add_begin_token: bool = True, name="codebook") -> Model:

    embedder = layers.Embedding(n_tokens, embd_shape.f_product, name=f"{name}/embd")

    def call(tokens):

        embd = embedder(tokens)

        return embd

    inputs = u.input_dict(
        Input(shape=embd_shape.s_shape, dtype=tf.int32, name="tokens"),
    )

    return Model(inputs=inputs, outputs=call(**inputs), name=name)

@export
class tokens(Enum):
    BEGIN = 0
    END   = 1

@export
def prepend_token(token: tokens, n_embd: int, name="prepend_token") -> Model:

    assert token in tokens, f"Unknown token {tokens!r}, must be one of {list(tokens.__members__)!r}"

    token_embedding = layers.Embedding(len(tokens), n_embd, name=f"{name}/begin_embd")

    tf_token = tf.constant(token.value, tf.int32)

    def call(embd):
        batch_size = shape(embd)[0]
        tokens = tf.tile([tf_token], [batch_size])
        token_embd = token_embedding(tokens)
        token_embd = ein.rearrange(token_embd, 'b embd -> b () embd')
        embd = tf.concat([token_embd, embd], axis=1) # concat along first sequence dim
        return embd

    inputs = u.input_dict(
        Input(shape=[None, n_embd], dtype=tf.float32, name="embd"),
    )

    return Model(inputs=inputs, outputs=call(**inputs), name=name)

@export
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

    inputs = u.input_dict(
        Input(shape=[], dtype=tf.int32, name="index"),
    )

    return Model(inputs=inputs, outputs=call(inputs), name="embd")

@export
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

    inputs = u.input_dict(
        Input(shape=input_shape.s_f_shape, dtype=tf.float32, name="angles"),
    )

    return Model(inputs=inputs, outputs=angle_call(**inputs), name=name)
