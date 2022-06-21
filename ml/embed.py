import math

import tensorflow as tf
from keras import layers, Model, Sequential

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

class Embedder(Model):
    
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embd_angle = Sequential([
            # layers.Dense(256),
            # layers.ReLU(),
            # layers.Dropout(cfg.dropout_rate),
            layers.Dense(cfg.embd_dim),
        ])
        # self.embd_angle = lambda x: tf.tile(x, [1, 1, cfg.embd_dim])
        self.embd_hand_idxs = layers.Embedding(cfg.n_hands, cfg.embd_dim)
        self.embd_frame_idxs = pos_enc(cfg.embd_dim, cfg.chunk_size*10)
        self.embd_dof_idxs = layers.Embedding(cfg.n_dof, cfg.embd_dim)
        self.embd_sentinel = layers.Embedding(1, cfg.embd_dim)
            
        self.unembed_layer = layers.Dense(1)

    def embed_single(self, inputs):

        # embed angles with a dense layer (add channel dim first)
        embd_angle = self.embd_angle(inputs["angles"][..., None])
        embd_frame_idxs = self.embd_frame_idxs(inputs["frame_idxs"])
        embd_hand_idxs = self.embd_hand_idxs(inputs["hand_idxs"])
        embd_dof_idxs = self.embd_dof_idxs(inputs["dof_idxs"])
        
        # concatenate the embeddings on the channel dim
        embd = tf.add_n([embd_angle, embd_hand_idxs, embd_frame_idxs, embd_dof_idxs])

        return embd

    def embed_sequence_with_begin_sentinel(self, inputs):

        embd = self.embed_single(inputs)

        sentinel_vec = self.embd_sentinel(tf.zeros_like(inputs["hand_idxs"][..., -1:]))
        
        # concatenate the BEGIN token on the seq dim
        embd = tf.concat([sentinel_vec, embd], axis=-2)

        return embd
    
    def unembed(self, embd):

        embd = self.unembed_layer(embd)
        
        # remove channel dim
        embd = embd[..., 0]

        return embd
