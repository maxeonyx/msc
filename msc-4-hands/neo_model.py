from sklearn.cluster import MiniBatchKMeans
import os
import socket
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model, Input, layers
from IPython.display import display
import tensorflow_datasets as tfds
import time
import matplotlib.pyplot as plt
import enlighten
import tensorflow_probability as tfp
from dotmap import DotMap
from icecream import ic

def create_look_backward_equal_mask(size_q, size_k):
    mask = 1 - tf.linalg.band_part(tf.ones((size_q, size_k)), -1, 0)
    return mask  # (size_q, size_k)

def create_look_backward_mask(size_q, size_k):
    mask = tf.linalg.band_part(tf.ones((size_q, size_k)), 0, -1)
    return mask  # (size_q, size_k)

def create_look_forward_mask(size_q, size_k):
    return tf.transpose(create_look_backward_mask(size_k, size_q))

def create_look_forward_equal_mask(size_q, size_k):
    return tf.transpose(create_look_backward_equal_mask(size_k, size_q))

# No masking. Use for evaluation when input and target have no overlap
MASK_NONE = tf.constant(0, dtype=tf.int32)
# Normal masking, each token only has access to previous tokens
MASK_BACKWARD = tf.constant(1, dtype=tf.int32)
# Use for non-offset self-attention, token has access to previous *and* itself
MASK_BACKWARD_EQUAL = tf.constant(2, dtype=tf.int32)
# Use inside target idx encoder
MASK_FORWARD = tf.constant(3, dtype=tf.int32)
# Use inside target idx encoder
MASK_FORWARD_EQUAL = tf.constant(4, dtype=tf.int32)

@tf.function
def get_mask(mask_type, seq_len_kv, seq_len_q):
    if mask_type == MASK_BACKWARD:
        return create_look_backward_mask(seq_len_q, seq_len_kv)
    elif mask_type == MASK_BACKWARD_EQUAL:
        return create_look_backward_equal_mask(seq_len_q, seq_len_kv)
    elif mask_type == MASK_FORWARD:
        return create_look_forward_mask(seq_len_q, seq_len_kv)
    elif mask_type == MASK_FORWARD_EQUAL:
        return create_look_forward_equal_mask(seq_len_q, seq_len_kv)
    else:
        return tf.zeros((seq_len_q, seq_len_kv))

def scaled_dot_product_attention(k, q, v, mask):
    batch_size = tf.shape(k)[0]
    seq_len_kv = tf.shape(k)[-2]
    kq_dim = tf.shape(k)[-1]
    seq_len_q = tf.shape(q)[-2]
    v_dim = tf.shape(v)[-1]
    
    dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # shape: (batch_size, n_heads, seq_len_q, seq_len_kv)
    
    dk = tf.cast(kq_dim, dtype=dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    scaled_attention_logits = tf.cast(scaled_attention_logits, tf.float32)
    scaled_attention_logits += mask * -1e9 # batch dim broadcast
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # sums to 1 along last axis
    # shape: (batch_size, seq_len_q, seq_len_kv)
    attention_weights = tf.cast(attention_weights, dtype)
    
    output = tf.matmul(attention_weights, v)
    # shape: (batch_size, seq_len_q, v_dim)
    
    return output, attention_weights

def multi_head_attention(m):
    embd_dim, n_heads = m.embd_dim, m.n_heads
    
    dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    
    wk = layers.Dense(embd_dim)
    wq = layers.Dense(embd_dim)
    wv = layers.Dense(embd_dim)
    dense = layers.Dense(embd_dim)
    
    assert embd_dim % n_heads == 0, "embd_dim must divide evenly into n_heads"
    head_width = embd_dim//n_heads
    
    def split_heads(x):
        xs = tf.shape(x)
        # reshape from (batch_size, seq_length, embd_dim) to (batch_size, num_heads, seq_len, head_width)
        x = tf.reshape(x, (xs[0], xs[1], n_heads, head_width))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(k, q, v, mask):
        batch_size = tf.shape(k)[0]
        
        k = wk(k)
        q = wq(q)
        v = wv(v)
        # shape: (batch_size, seq_len_*, embd_dim)
        
        k = split_heads(k)
        q = split_heads(q)
        v = split_heads(v)
        # shape: (batch_size, num_heads, seq_len_*, head_width)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(k, q, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len, num_heads, depth)
        output = tf.reshape(scaled_attention, (batch_size, -1, embd_dim))
        # output = dense(output)
        return output, attention_weights
    return call

def pointwise_feedforward_layer(m, hidden_dim, output_layer, n_hidden_layers=1, dtype=None):
    hidden_layers = [layers.Dense(hidden_dim) for _ in range(n_hidden_layers)]
    dtype = dtype or tf.keras.mixed_precision.global_policy()
    
    def call(x):
        for layer in hidden_layers:
            x = layer(x)
            x = m.activation_fn(x)
        x = output_layer(x)
        return x
    return call


def transformer_layer(m):
    mha = multi_head_attention(m)
    ffl = pointwise_feedforward_layer(m, m.ffl_dim, output_layer=layers.Dense(m.embd_dim))
    layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    dropout1 = layers.Dropout(m.dropout_rate)
    dropout2 = layers.Dropout(m.dropout_rate)
    def call(kv_embd, q_embd, mask):
        x = q_embd
        out1 = layernorm1(dropout1(kv_embd)) # prenorm
        attn_out, attn_weights = mha(out1, q_embd, out1, mask)
        x += attn_out

        out2 = layernorm2(dropout2(x)) # prenorm
        ffl_out = ffl(out2)
        x += ffl_out

        return x
    return call


class ConditionalGenerativeTransformer(tf.keras.Model):

    def __init__(self, ):
        super().__init__()

        self.enc_a_layers = [transformer_layer(m) for _ in range(m.n_enc_a_layers)]
        self.enc_b_layers = [transformer_layer(m) for _ in range(m.n_enc_b_layers)]
        self.dec_layer = transformer_3sep_layer(m)
        self.x_encoder = pointwise_feedforward_layer(m, m.ffl_dim, output_layer=layers.Dense(m.embd_dim))
        self.final_dropout = layers.Dropout(m.dropout_rate)
        self.final_layer_norm = layers.LayerNormalization(epsilon=1e-6)
        
        self.decoder = pointwise_feedforward_layer(m, m.dec_dim, output_layer=output_layer, n_hidden_layers=m.n_dec_layers)
