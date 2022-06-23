import math

import tensorflow as tf
from keras import layers, Model

from ml import util

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
        encoding = util.angle_wrap(vals * scale)
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

class AngleOnlyEmbedder(Model):
    
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embd_angle = angle_pos_enc(cfg, cfg.embd_dim)
        self.embd_sentinel = layers.Embedding(1, cfg.embd_dim)

    def embed_single(self, inputs):

        embd_angle = self.embd_angle(inputs["angles"])

        return embd_angle

    def embed_sequence_with_begin_sentinel(self, inputs, length=1):

        embd = self.embed_single(inputs)

        batch_dims = tf.shape(embd)[:-2]

        shape = tf.concat([batch_dims, [length]], axis=0)

        sentinel_vec = self.embd_sentinel(tf.zeros(shape, dtype=tf.int32))
        
        # concatenate the BEGIN token on the seq dim
        embd = tf.concat([sentinel_vec, embd], axis=-2)

        return embd

class AllIndicesConcatEmbedder(Model):
    
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embd_angle = angle_pos_enc(cfg, cfg.embd_dim // 4)
        # self.embd_angle = lambda x: tf.tile(x, [1, 1, cfg.embd_dim])
        self.embd_hand_idxs = layers.Embedding(cfg.n_hands, cfg.embd_dim // 4)
        self.embd_frame_idxs = pos_enc(cfg.embd_dim // 4, 10000)
        self.embd_dof_idxs = layers.Embedding(cfg.n_dof, cfg.embd_dim // 4)
        self.embd_sentinel = layers.Embedding(1, cfg.embd_dim)

    def embed_single(self, inputs):

        embd_angle = self.embd_angle(inputs["angles"])
        embd_frame_idxs = self.embd_frame_idxs(inputs["frame_idxs"])
        embd_hand_idxs = self.embd_hand_idxs(inputs["hand_idxs"])
        embd_dof_idxs = self.embd_dof_idxs(inputs["dof_idxs"])
        
        # concatenate the embeddings on the channel dim
        embd = tf.concat([embd_angle, embd_hand_idxs, embd_frame_idxs, embd_dof_idxs], axis=-1)

        return embd

    def embed_sequence_with_begin_sentinel(self, inputs, length=1):

        embd = self.embed_single(inputs)

        batch_dims = tf.shape(embd)[:-2]

        shape = tf.concat([batch_dims, [length]], axis=0)

        sentinel_vec = self.embd_sentinel(tf.zeros(shape, dtype=tf.int32))
        
        # concatenate the BEGIN token on the seq dim
        embd = tf.concat([sentinel_vec, embd], axis=-2)

        return embd

class AllIndicesAddEmbedder(Model):
    
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embd_angle = angle_pos_enc(cfg, cfg.embd_dim)
        # self.embd_angle = lambda x: tf.tile(x, [1, 1, cfg.embd_dim])
        self.embd_hand_idxs = layers.Embedding(cfg.n_hands, cfg.embd_dim)
        self.embd_frame_idxs = pos_enc(cfg.embd_dim, 10000)
        self.embd_dof_idxs = layers.Embedding(cfg.n_dof, cfg.embd_dim)
        self.embd_sentinel = layers.Embedding(1, cfg.embd_dim)

    def embed_single(self, inputs):

        embd_angle = self.embd_angle(inputs["angles"])
        embd_frame_idxs = self.embd_frame_idxs(inputs["frame_idxs"])
        embd_hand_idxs = self.embd_hand_idxs(inputs["hand_idxs"])
        embd_dof_idxs = self.embd_dof_idxs(inputs["dof_idxs"])
        
        # concatenate the embeddings on the channel dim
        embd = tf.add_n([embd_angle, embd_hand_idxs, embd_frame_idxs, embd_dof_idxs])

        return embd

    def embed_sequence_with_begin_sentinel(self, inputs, length=1):

        embd = self.embed_single(inputs)

        batch_dims = tf.shape(embd)[:-2]

        shape = tf.concat([batch_dims, [length]], axis=0)

        sentinel_vec = self.embd_sentinel(tf.zeros(shape, dtype=tf.int32))
        
        # concatenate the BEGIN token on the seq dim
        embd = tf.concat([sentinel_vec, embd], axis=-2)

        return embd
