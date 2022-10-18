from enum import Enum

from mx.prelude import *
from ._blocks import featurewise_dense
from mx.utils import Einshape, tf_scope

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

    initializer = tf.keras.initializers.TruncatedNormal(stddev=1/sqrt(n_embd))

    token_embedding = layers.Embedding(len(tokens), n_embd, embeddings_initializer=initializer, name=f"{name}/begin_embd")

    tf_token = tf.constant(token.value, tf.int32)

    def call(input):
        embd = inputs["embd"]
        batch_size = shape(embd)[0]

        tokens = tf.repeat(tf_token[None], batch_size[None])
        tokens = tokens[:, None]
        token_embd = token_embedding(tokens)
        embd = tf.concat([token_embd, embd], axis=1) # concat along first sequence dim
        return embd

    inputs = u.input_dict(
        Input(shape=[None, n_embd], dtype=u.dtype(), name="embd"),
    )

    return Model(inputs=inputs, outputs=call(inputs), name=name)

@export
def positional_embedding(n_embd, max_wavelength=10000, name="embd") -> Model:
    assert n_embd % 2 == 0, f"embd_dim must be divisible by 2 to use positional encoding, got embd_dim={n_embd}"

    # based from the keras source code
    # https://github.com/keras-team/keras-nlp/blob/v0.3.0/keras_nlp/layers/sine_position_encoding.py#L21

    @tf_scope
    def positional_encoding(inputs):
        idxs = inputs["idxs"]
        position = tf.cast(idxs, u.dtype())
        min_freq = 1. / max_wavelength
        timescales = tf.pow(
            min_freq,
            tf.range(n_embd, dtype=tf.float32) / n_embd
        )
        timescales = tf.cast(timescales, u.dtype())
        position = ein.rearrange(position, '... seq -> ... seq ()')
        timescales = ein.rearrange(timescales, '... embd -> ... () embd')
        angles = position * timescales
        # even indices are sine, odd are cosine
        cos_mask = tf.cast(tf.range(n_embd) % 2, u.dtype())
        sin_mask = 1 - cos_mask
        # embedding shape is [seq_length, hidden_size]
        positional_encodings = (
            tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
        )

        # scale norm. because we use sin/cos we scale by 1/sqrt(D/2) instead of 1/sqrt(D)
        positional_encodings *= tf.cast(tf.math.sqrt(tf.cast(n_embd // 2, u.dtype())), u.dtype())

        return positional_encodings

    inputs = u.input_dict(
        Input(shape=[None], dtype=tf.int32, name="idxs"),
    )

    return Model(inputs=inputs, outputs=positional_encoding(inputs), name=name)

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
        offsets = tf.range(num_repeats, dtype=u.dtype()) * scale
        # add "repeats" dim
        angles = angles[..., None]
        angles = angles + tf.broadcast_to(offsets, tf.broadcast_dynamic_shape(tf.shape(offsets), tf.shape(angles)))
        # add "sincos" dim
        angles = tf.stack([tf.sin(angles), tf.cos(angles)], axis=-1)
        # flatten to "embd" dim
        embd = dense_out(angles)

        return embd

    inputs = u.input_dict(
        Input(shape=input_shape.s_f_shape, dtype=u.dtype(), name="angles"),
    )

    return Model(inputs=inputs, outputs=angle_call(**inputs), name=name)
