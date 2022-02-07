from sklearn.cluster import MiniBatchKMeans
import os
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

def model_name(hostname, config):
    
    spec = []
    
    if config.ds != 'mnist':
        spec.append(config.dataset.name)
        
    if config.dataset.rescale is not None:
        spec.append(f'{config.dataset.image_size[0]}x{config.dataset.image_size[1]}')
    
    if config.dataset.n_colors != 4:
        spec.append(f'n{config.dataset.n_colors}')
    
    if config.noise_fraction != 0:
        spec.append(f'noise{config.noise_fraction}')
        
    # batch size triple
    if config.grad_accum_steps is None:
        accum_steps = '1'
    elif type(config.grad_accum_steps) is int:
        accum_steps = str(config.grad_accum_steps)
    else:
        accum_steps = 'DYN'
    spec.append(f'bs{config.num_devices}x{accum_steps}x{config.minibatch_size}')
    
    return f"{hostname}-{'-'.join(spec)}"


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
    
def print_masks():
    print("backward")
    print(create_look_backward_mask(7,7))
    print()
    print("backward_equal")
    print(create_look_backward_equal_mask(7,7))
    print()
    print("forward")
    print(create_look_forward_mask(7,7))
    print()
    print("forward_equal")
    print(create_look_forward_equal_mask(7,7))
    print()



# scale is the max-min of vals
# for mnist it's 28 because thats the width and height of the images
def dual_positional_encoding(n_dims, length):
    one_axis_dim = n_dims//2
    i = tf.range(n_dims//4, dtype=tf.float32)
    i = tf.expand_dims(i, -2)
    scale = tf.pow(length, 2.*i/one_axis_dim)
    
    def pos_enc(vals):
        vals = tf.expand_dims(vals, -1)

        # the bit inside the sin / cos
        rate = vals / scale
        sin = tf.sin(rate)
        cos = tf.cos(rate)
        encoding = tf.concat([sin, cos], axis=-1)
        return encoding
    
    def call(idxs):
        rows = idxs // 28
        cols = idxs % 28
        
        row_enc = pos_enc(rows)
        col_enc = pos_enc(cols)
        
        encoding = tf.concat([row_enc, col_enc], axis=-1)
        return encoding
        
    return call

def linear_position_encoding(dim_lengths, out_vec_lengths):
    def call(inp):        
        dims_enc = None
        for dim in range(len(dim_lengths)):
            enc = inp
            for dim_len in dim_lengths[dim+1:]:
                enc = enc // dim_len
            enc = enc % dim_lengths[dim]
            enc = tf.cast(enc, tf.float32) / tf.cast(dim_lengths[dim] - 1, tf.float32)
            enc = tf.expand_dims(enc, -1)
            tile_shape = [1 for _ in enc.shape[:-1]] + [out_vec_lengths[dim]]
            enc = tf.tile(enc, tile_shape)
            dims_enc = enc if dims_enc is None else tf.concat([dims_enc, enc], axis=-1)
        
        return dims_enc
        
    return call

def scaled_dot_product_attention(k, q, v, mask):
    batch_size = tf.shape(k)[0]
    seq_len_kv = tf.shape(k)[-2]
    kq_dim = tf.shape(k)[-1]
    seq_len_q = tf.shape(q)[-2]
    v_dim = tf.shape(v)[-1]
    
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # shape: (batch_size, n_heads, seq_len_q, seq_len_kv)
    
    dk = tf.cast(kq_dim, tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    scaled_attention_logits += mask * -1e9 # batch dim broadcast
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # sums to 1 along last axis
    # shape: (batch_size, seq_len_q, seq_len_kv)
    
    output = tf.matmul(attention_weights, v)
    # shape: (batch_size, seq_len_q, v_dim)
    
    return output, attention_weights

# # position-content disentagled attention from DeBERTa
# # https://arxiv.org/abs/2006.03654
# def deberta_attention(m):
#     wk_content = layers.Dense(embd_dim)
#     wq_content = layers.Dense(embd_dim)
#     wv_content = layers.Dense(embd_dim)
#     wk_position = layers.Dense(embd_dim)
#     wq_position = layers.Dense(embd_dim)
#     dense = layers.Dense(embd_dim)
    
#     assert embd_dim % n_heads == 0, "embd_dim must divide evenly into n_heads"
#     head_width = embd_dim//n_heads
    
#     def split_heads(x, batch_size):
#         # reshape from (batch_size, seq_length, embd_dim) to (batch_size, num_heads, seq_len, head_width)
#         x = tf.reshape(x, (batch_size, -1, n_heads, head_width))
#         return tf.transpose(x, perm=[0, 2, 1, 3])
    
#     def mha(k, q, v, mask):
#         batch_size = tf.shape(k)[0]
        
#         k = wk(k)
#         q = wk(q)
#         v = wk(v)
#         # shape: (batch_size, seq_len_*, embd_dim)
        
#         k = split_heads(k, batch_size)
#         q = split_heads(q, batch_size)
#         v = split_heads(v, batch_size)
#         # shape: (batch_size, num_heads, seq_len_*, head_width)
        
#         scaled_attention, attention_weights = scaled_dot_product_attention(k, q, v, mask)
#         scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
#         # (batch_size, seq_len, num_heads, depth)
#         concat_attention = tf.reshape(scaled_attention, (batch_size, -1, embd_dim))
#         output = dense(concat_attention)
#         return output, attention_weights
    
#     def call_self_attn(x, pos):
        
    
#     return call

def multi_head_attention(embd_dim, n_heads):
    
    wk = layers.Dense(embd_dim)
    wq = layers.Dense(embd_dim)
    wv = layers.Dense(embd_dim)
    dense = layers.Dense(embd_dim)
    
    assert embd_dim % n_heads == 0, "embd_dim must divide evenly into n_heads"
    head_width = embd_dim//n_heads
    
    def split_heads(x, batch_size):
        # reshape from (batch_size, seq_length, embd_dim) to (batch_size, num_heads, seq_len, head_width)
        x = tf.reshape(x, (batch_size, -1, n_heads, head_width))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(k, q, v, mask):
        batch_size = tf.shape(k)[0]
        
        k = wk(k)
        q = wq(q)
        v = wv(v)
        # shape: (batch_size, seq_len_*, embd_dim)
        
        k = split_heads(k, batch_size)
        q = split_heads(q, batch_size)
        v = split_heads(v, batch_size)
        # shape: (batch_size, num_heads, seq_len_*, head_width)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(k, q, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, embd_dim))
        output = dense(concat_attention)
        return output, attention_weights
    return call
    
def pointwise_feedforward_layer(m, hidden_dim, out_dim, n_hidden_layers=1):
    hidden_layers = [layers.Dense(hidden_dim) for _ in range(n_hidden_layers)]
    dense2 = layers.Dense(out_dim)
    
    def call(x):
        for layer in hidden_layers:
            x = layer(x)
            x = m.activation_fn(x)
        x = dense2(x)
        return x
    return call


def transformer_layer(m):
    mha = multi_head_attention(m.embd_dim, m.n_heads)
    ffl = pointwise_feedforward_layer(m, m.ffl_dim, m.embd_dim)
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
    
def transformer_3sep_layer(m):
    mha = multi_head_attention(m.embd_dim, m.n_heads)
    ffl = pointwise_feedforward_layer(m, m.ffl_dim, m.embd_dim)
    layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    dropout1 = layers.Dropout(m.dropout_rate)
    dropout2 = layers.Dropout(m.dropout_rate)
    def call(key, query, val, mask):
        key = layernorm1(dropout1(key)) # prenorm
        val = layernorm1(dropout1(val)) # prenorm
        attn_out, attn_weights = mha(key, query, val, mask)
        x = query + attn_out

        out2 = layernorm2(dropout2(x)) # prenorm
        ffl_out = ffl(out2)
        x += ffl_out

        return x
    return call
    
# attentive neural process without global latent
def anp_architecture_no_global_latent(m):
    enc_a_layers = [transformer_layer(m) for _ in range(m.n_enc_a_layers)]
    dec_layer = transformer_3sep_layer(m)
    x_encoder = pointwise_feedforward_layer(m, m.ffl_dim, m.embd_dim)
    final_dropout = layers.Dropout(m.dropout_rate)
    final_layer_norm = layers.LayerNormalization(epsilon=1e-6)
    final_layer = pointwise_feedforward_layer(m, m.dec_dim, m.n_colors, n_hidden_layers=m.n_dec_layers)
    def call(inp_xy, inp_x, tar_x, enc_a_mask, dec_mask):
        inp_x = x_encoder(inp_x)
        tar_x = x_encoder(tar_x)
        for enc_layer in enc_a_layers:
            inp_xy += enc_layer(inp_xy, inp_xy, mask=enc_a_mask)
        tar_y = dec_layer(key=inp_x, query=tar_x, val=inp_xy, mask=dec_mask)
        tar_y = final_layer_norm(final_dropout(tar_y))
        outs = final_layer(tar_y)
        return outs
    return call
    
# attentive neural process
def anp_architecture(m):
    enc_a_layers = [transformer_layer(m) for _ in range(m.n_enc_a_layers)]
    enc_b_layers = [transformer_layer(m) for _ in range(m.n_enc_b_layers)]
    dec_layer = transformer_3sep_layer(m)
    x_encoder = pointwise_feedforward_layer(m, m.ffl_dim, m.embd_dim)
    final_dropout = layers.Dropout(m.dropout_rate)
    final_layer_norm = layers.LayerNormalization(epsilon=1e-6)
    final_layer = pointwise_feedforward_layer(m, m.dec_dim, m.n_colors, n_hidden_layers=m.n_dec_layers)
    def call(inp_xy, inp_x, tar_x, enc_a_mask, dec_mask):
        inp_x = x_encoder(inp_x)
        tar_x = x_encoder(tar_x)
        enc_inp_a = inp_xy
        for enc_layer in enc_a_layers:
            enc_inp_a += enc_layer(enc_inp_a, enc_inp_a, mask=enc_a_mask)
        enc_inp_b = inp_xy
        for enc_layer in enc_a_layers:
            enc_inp_b += enc_layer(enc_inp_b, enc_inp_b, mask=enc_a_mask)
            
        tar_z = dec_layer(key=inp_x, query=tar_x, val=inp_xy, mask=dec_mask)
        tar_z = final_layer_norm(final_dropout(tar_z))
        
        global_latent = tf.reduce_mean(enc_inp_b, axis=1) # mean along sequence dim
        global_latent = tf.expand_dims(global_latent, 1) # add middle dim back for broadcasting
        global_latent *= tf.ones_like(tar_z)
        
        tar_z = tf.concat([tar_z, global_latent], axis=2) # concat along embd dim
        tar_y = final_layer(tar_z)
        return tar_y
    return call

# custom attentive neural process
def canp_architecture(m):
    enc_a_layers = [transformer_layer(m) for _ in range(m.n_enc_a_layers)]
    dec_layer = transformer_layer(m)
    final_dropout = layers.Dropout(m.dropout_rate)
    final_layer_norm = layers.LayerNormalization(epsilon=1e-6)
    final_layer = pointwise_feedforward_layer(m.ffl_dim, m.n_colors)
    def call(xa, xb, enc_a_mask, dec_mask):
        for enc_layer in enc_a_layers:
            xa += enc_layer(xa, xa, mask=enc_a_mask)
        tar_y = dec_layer(xa, xb, mask=dec_mask)
        tar_y = final_layer_norm(final_dropout(tar_y))
        outs = final_layer(tar_y)
        return outs
    return call

def relative_position_embedding(pos_a, pos_b):
    # assume pos_a and pos_b are indices
    pos_a = tf.expand_dims(pos_a, -2)
    pos_b = tf.expand_dims(pox_b, -1)
    
    relative_pos = pos_a - pos_b
    

def transformer(m):
    
    colors = Input([None])
    inp_idxs = Input([None])
    tar_idxs = Input([None])
    # use type_spec argument because we don't want the implicit batch dim for these inputs
    enc_a_mask = Input(type_spec=tf.TensorSpec(shape=[None, None]))
    dec_mask = Input(type_spec=tf.TensorSpec(shape=[None, None]))
    
    
    if m.activation == 'relu':
        m.activation_fn = tf.nn.relu
    elif m.activation == 'swish':
        m.activation_fn = tf.nn.silu
    elif m.activation == 'gelu':
        m.activation_fn = tf.nn.gelu
    
    
    col_embd = layers.Embedding(m.n_colors, m.embd_dim)(colors)
    
    if m.position_embedding == 'pos_enc':
        # set length to somewhat bigger than the image width/height
        position_embedding = dual_positional_encoding(n_dims=m.embd_dim, length=100)
    if m.position_embedding == 'learned':
        # learned vector codebook embedding
        position_embedding = layers.Embedding(m.seq_len, m.embd_dim)
    if m.position_embedding == 'linear':
        # linear mapping from idx to 0..1 range for each position dimension
        position_embedding = linear_position_encoding(m.image_size, [m.embd_dim // 2, m.embd_dim - (m.embd_dim // 2)])
    if m.position_embedding == 'linear_learned':
        lin_embd = linear_position_encoding(m.image_size, [1, 1])
        position_embedding = lambda x_inp: layers.Dense(m.embd_dim)(layers.Dense(m.embd_dim, activation=m.activation_fn)(lin_embd(x_inp)))
            
    
    inp_pos_embd = position_embedding(inp_idxs)
    tar_pos_embd = position_embedding(tar_idxs)
    
    xa = col_embd + inp_pos_embd
    xb = tar_pos_embd
    
    if m.architecture == 'anp':
        output_layer = anp_architecture(m)(xa, inp_pos_embd, tar_pos_embd,  enc_a_mask, dec_mask)
    
    
    return Model(inputs=[colors, inp_idxs, tar_idxs, enc_a_mask, dec_mask], outputs=[output_layer])

