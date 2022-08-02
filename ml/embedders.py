import math
import typing

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import Input, Model, layers
import einops

from ml import utils

def pos_enc(n_dims, scale):
    assert n_dims % 2 == 0, "n_dims must be divisible by 2 to use positional encoding"
    dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    
    i = tf.range(n_dims//2)
    i = tf.cast(i, dtype=dtype)
    i = tf.expand_dims(i, -2)
    
    scale = tf.pow(scale, 2.*i/n_dims) 
    
    def call(vals):
        vals = tf.expand_dims(vals, -1)
        # the bit inside the sin / cos
        rate = math.pi*tf.cast(vals, dtype=dtype) / scale
        sin = tf.sin(rate)
        cos = tf.cos(rate)
        encoding = tf.concat([sin, cos], axis=-1)
        return encoding
    
    return call

def circular_pos_enc(cfg, n_dims):
    """
    Positional encoding where the input is already an angle.
    Only uses integer scales, so that the encoding of an angle is unique.
    """
    
    i = tf.range(n_dims)
    i = tf.cast(i, dtype=tf.float32)
    scale = tf.round(tf.pow(1.2, i))
    scale = tf.expand_dims(scale, -2)
    
    def call(vals):
        vals = tf.expand_dims(vals, -1)
        encoding = utils.angle_wrap(vals * scale)
        if cfg.scale_to_1_1:
            encoding /= math.pi
        return encoding
    
    return call

def angle_pos_enc(cfg, n_dims):
    assert n_dims % 2 == 0, "n_dims must be divisible by 2 to use angle_pos_enc"
    
    i = tf.range(1, n_dims//2 + 1)
    i = tf.cast(tf.expand_dims(i, -2), tf.float32)
    
    def call(vals):
        vals = tf.expand_dims(vals, -1)
        sin = tf.sin(vals*i)
        cos = tf.cos(vals*i)
        encoding = tf.concat([sin, cos], axis=-1)
        return encoding
    
    return call

def add_embedder(cfg, name="embedder"):
    angles = Input(shape=[None], dtype=tf.float32, name="angles")
    frame_idxs = Input(shape=[None], dtype=tf.int32, name="frame_idxs")
    hand_idxs = Input(shape=[None], dtype=tf.int32, name="hand_idxs")
    dof_idxs = Input(shape=[None], dtype=tf.int32, name="dof_idxs")

    batch_size = tf.shape(angles)[0]

    embd_angle = angle_pos_enc(cfg, cfg.embd_dim)(angles)
    embd_frame_idxs = pos_enc(cfg.embd_dim, 10000)(frame_idxs)
    embd_hand_idxs = layers.Embedding(cfg.n_hands, cfg.embd_dim)(hand_idxs)
    embd_dof_idxs = layers.Embedding(cfg.n_dof, cfg.embd_dim)(dof_idxs)

    embd = tf.add_n([embd_angle, embd_frame_idxs, embd_hand_idxs, embd_dof_idxs])

    begin_tok = layers.Embedding(1, cfg.embd_dim)(tf.zeros([batch_size, 1], dtype=tf.int32))
    embd = tf.concat([begin_tok, embd], axis=1) # concat on frame/sequence dim

    return Model(inputs=[angles, frame_idxs, hand_idxs, dof_idxs], outputs=embd, name=name)
    
