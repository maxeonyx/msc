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


def model_name(config):
    
    spec = []
    
    hostname = socket.gethostname().split(".")[0]
    
    if config.ds != 'mnist':
        spec.append(config.ds)
        
    if 'rescale' in config.dataset:
        spec.append(f'{config.dataset.image_size[0]}x{config.dataset.image_size[1]}')
    
    if config.dataset.discrete and config.dataset.n_colors != 4:
        spec.append(f'n{config.dataset.n_colors}')
        
    if config.dataset.shuffle == False:
        spec.append('noshuf')
    
    if config.dataset.continuous:
        spec.append(f'contin')
    
    if config.dataset.noise_fraction is not None:
        spec.append(f'noise{config.dataset.noise_fraction}')
        
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



# position-content disentagled attention from DeBERTa
# https://arxiv.org/abs/2006.03654

def deberta_attention(m):
    embd_dim, n_heads = m.embd_dim, m.n_heads
    wk_content = layers.Dense(embd_dim)
    wq_content = layers.Dense(embd_dim)
    wv_content = layers.Dense(embd_dim)
    wk_position = layers.Dense(embd_dim)
    wq_position = layers.Dense(embd_dim)
    dense = layers.Dense(embd_dim)
    
    use_relative_positions = m.use_relative_positions
    
    dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    
    assert embd_dim % n_heads == 0, "embd_dim must divide evenly into n_heads"
    head_width = embd_dim//n_heads
    
    def split_heads(x):
        with tf.name_scope("split_heads") as scope:
            xs = tf.shape(x)
            # reshape from (batch_size, seq_length, embd_dim) to (batch_size, num_heads, seq_len, head_width)
            x = tf.reshape(x, (xs[0], xs[1], n_heads, head_width))
            x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x
    
    def split_heads_r_matrix(x):
        with tf.name_scope("split_heads_r_matrix") as scope:
            xs = tf.shape(x)
            # reshape from (batch_size, q_len, k_len, embd_dim) to (batch_size, q_len, k_len, num_heads, head_width)
            x = tf.reshape(x, (xs[0], xs[1], xs[2], n_heads, head_width))

            # reshape from (batch_size, q_len, k_len, num_heads, head_width) to (batch_size, num_heads, q_len, k_len, head_width)
            x = tf.transpose(x, perm=[0, 3, 1, 2, 4])
        return x
    
    # @tf.function
    def call(k, q, v, pos_k, pos_q, mask=None):
        
        with tf.name_scope("deberta_mha") as scope:

            batch_size = tf.shape(k)[0]

            k = wk_content(k)
            q = wq_content(q)
            v = wv_content(v)
            # shape: (batch_size, seq_len_*, embd_dim)

            pos_k = wk_position(pos_k)
            pos_q = wq_position(pos_q)
            # shape: (batch_size, seq_len, seq_len, embd_dim)

            # split attention heads
            k = split_heads(k)
            q = split_heads(q)
            v = split_heads(v)
            
            if use_relative_positions:
                pos_k = split_heads_r_matrix(pos_k)
                pos_q = split_heads_r_matrix(pos_q)
            else:
                pos_k = split_heads(pos_k)
                pos_q = split_heads(pos_q)
                # pos_k = tf.expand_dims(pos_k, axis=-3)
                # pos_q = tf.expand_dims(pos_q, axis=-2)
            # shape: (batch_size, num_heads, seq_len_*, head_width)

            #scaled_attention, attention_weights = scaled_dot_product_attention(k, q, v, mask)
            ####

            batch_size = tf.shape(k)[0]
            seq_len_kv = tf.shape(k)[-2]
            kq_dim = tf.shape(k)[-1]
            seq_len_q = tf.shape(q)[-2]
            v_dim = tf.shape(v)[-1]

            
            attention_scale_factor = 0.
            attention_components = []
            
            ### Content/Content
            attention_scale_factor += 1.
            attention_components.append(tf.matmul(q, k, transpose_b=True))

            ### Content/Position
            attention_scale_factor += 1.
            attention_components.append(tf.matmul(q, pos_k, transpose_b=True))

            ### Position/Content
            attention_scale_factor += 1.
            attention_components.append(tf.matmul(pos_q, k, transpose_b=True))
            # shape: (batch_size, n_heads, seq_len_q, seq_len_kv)
            
            if not use_relative_positions:
                ### Position/Position
                attention_scale_factor += 1.
                attention_components.append(tf.matmul(pos_q, pos_k, transpose_b=True))
            
            attention_logits = tf.add_n(attention_components, name='attention_logits')

            dk = tf.cast(kq_dim, dtype=dtype)
            scaled_attention_logits = attention_logits / tf.math.sqrt(attention_scale_factor * dk)
            scaled_attention_logits = tf.cast(scaled_attention_logits, tf.float32)
            scaled_attention_logits += tf.multiply(mask, -1e9, name='bigmul') # batch dim broadcast

            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # sums to 1 along last axis
            attention_weights = tf.cast(attention_weights, dtype)
            # shape: (batch_size, seq_len_q, seq_len_kv)

            scaled_attention = tf.matmul(attention_weights, v)
            # shape: (batch_size, seq_len_q, v_dim)

            # recombine heads
            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
            # (batch_size, seq_len, num_heads, depth)
            concat_attention = tf.reshape(scaled_attention, (batch_size, -1, embd_dim))
            # (batch_size, seq_len, embd_dim)
        
        
        output = concat_attention
        # output = dense(concat_attention)
        
        return output, attention_weights
    
    return call

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
        if output_layer is not None:
            x = output_layer(x)
        return x
    return call


def deberta_layer(m):
    mha = deberta_attention(m)
    ffl = pointwise_feedforward_layer(m, m.ffl_dim, output_layer=layers.Dense(m.embd_dim))
    layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    dropout1 = layers.Dropout(m.dropout_rate)
    dropout2 = layers.Dropout(m.dropout_rate)
    def call(keyval, query, pos_q, pos_k, mask):
        x = query
        out1 = layernorm1(dropout1(keyval)) # prenorm
        attn_out, attn_weights = mha(q=x, k=out1, v=out1, pos_q=pos_q, pos_k=pos_k, mask=mask)
        x += attn_out

        out2 = layernorm2(dropout2(x)) # prenorm
        ffl_out = ffl(out2)
        x += ffl_out

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
    
def transformer_3sep_layer(m):
    mha = multi_head_attention(m)
    ffl = pointwise_feedforward_layer(m, m.ffl_dim, output_layer=layers.Dense(m.embd_dim))
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

    
# deberta / anp
def deberta_anp_architecture(m):
    enc_a_layers = [deberta_layer(m) for _ in range(m.n_enc_a_layers)]
    enc_b_layers = [deberta_layer(m) for _ in range(m.n_enc_b_layers)]
    dec_layer = deberta_layer(m)
    x_encoder = pointwise_feedforward_layer(m, m.ffl_dim, output_layer=layers.Dense(m.embd_dim))
    # final_dropout = layers.Dropout(m.dropout_rate)
    # final_layer_norm = layers.LayerNormalization(epsilon=1e-6)
    # final_layer = pointwise_feedforward_layer(m, m.dec_dim, m.output_dim, n_hidden_layers=m.n_dec_layers)
    decoder_dtype = tf.keras.mixed_precision.Policy("float32")
    decoder = pointwise_feedforward_layer(m, m.dec_dim, output_layer=None, n_hidden_layers=m.n_dec_layers, dtype=decoder_dtype)
    def call(inp, inp_x, tar_x, r_self, r_cross, enc_mask, dec_mask):
        
        if m.use_relative_positions:
            pos_q_self = r_self
            pos_k_self = r_self
            pos_q_cross = r_cross
            pos_k_cross = r_cross
        else:
            pos_q_self = inp_x
            pos_k_self = inp_x
            pos_q_cross = tar_x
            pos_k_cross = inp_x
        
        inp_x = x_encoder(inp_x)
        enc_a_z = inp_x
        for enc_layer in enc_a_layers:
            enc_a_z += enc_layer(keyval=enc_a_z, query=enc_a_z, pos_q=pos_q_self, pos_k=pos_k_self, mask=enc_mask)
        enc_b_z = inp
        for enc_layer in enc_b_layers:
            enc_b_z += enc_layer(keyval=enc_b_z, query=enc_b_z, pos_q=pos_q_self, pos_k=pos_k_self, mask=enc_mask)
        
        tar_z = x_encoder(tar_x)
        tar_z = dec_layer(keyval=enc_a_z, query=tar_z, pos_q=pos_q_cross, pos_k=pos_k_cross, mask=dec_mask)
        # tar_z = final_layer_norm(final_dropout(tar_z))
        
        global_latent = tf.reduce_mean(enc_b_z, axis=1) # mean along sequence dim
        global_latent = tf.expand_dims(global_latent, 1) # add middle dim back for broadcasting
        global_latent = tf.multiply(global_latent, tf.ones_like(tar_z), name='broadcast_global_latent')
        
        tar_z = tf.concat([tar_z, global_latent], axis=2) # concat along embd dim
        
        # decoder here
        tar_y = decoder(tar_z)
        
        return tar_y
    return call
    
# attentive neural process
def anp_architecture(m):
    enc_a_layers = [transformer_layer(m) for _ in range(m.n_enc_a_layers)]
    enc_b_layers = [transformer_layer(m) for _ in range(m.n_enc_b_layers)]
    dec_layer = transformer_3sep_layer(m)
    x_encoder = pointwise_feedforward_layer(m, m.ffl_dim, output_layer=layers.Dense(m.embd_dim))
    final_dropout = layers.Dropout(m.dropout_rate)
    final_layer_norm = layers.LayerNormalization(epsilon=1e-6)
    decoder_dtype = tf.keras.mixed_precision.Policy("float32")
    decoder = pointwise_feedforward_layer(m, m.dec_dim, output_layer=None, n_hidden_layers=m.n_dec_layers, dtype=decoder_dtype)
    def call(inp_xy, inp_x, tar_x, enc_a_mask, dec_mask):
        inp_x = x_encoder(inp_x)
        tar_x = x_encoder(tar_x)
        enc_inp_a = inp_xy
        for enc_layer in enc_a_layers:
            enc_inp_a += enc_layer(enc_inp_a, enc_inp_a, mask=enc_a_mask)
        enc_inp_b = inp_xy
        for enc_layer in enc_b_layers:
            enc_inp_b += enc_layer(enc_inp_b, enc_inp_b, mask=enc_a_mask)
            
        tar_z = dec_layer(key=enc_inp_a, query=tar_x, val=enc_inp_a, mask=dec_mask)
        tar_z = final_layer_norm(final_dropout(tar_z))
        
        global_latent = tf.reduce_mean(enc_inp_b, axis=1) # mean along sequence dim
        global_latent = tf.expand_dims(global_latent, 1) # add middle dim back for broadcasting
        global_latent *= tf.ones_like(tar_z)
        
        tar_z = tf.concat([tar_z, global_latent], axis=2) # concat along embd dim
        tar_y = decoder(tar_z)
        return tar_y
    return call



def linear_position_encoding(dim_lengths, out_vec_lengths):
    dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    def call(inp):        
        dims_enc = None
        for dim in range(len(dim_lengths)):
            enc = inp
            for dim_len in dim_lengths[dim+1:]:
                enc = enc // dim_len
            enc = enc % dim_lengths[dim]
            enc = tf.cast(enc, dtype=dtype) / tf.cast(dim_lengths[dim] - 1, dtype=dtype)
            enc = tf.expand_dims(enc, -1)
            tile_shape = [1 for _ in enc.shape[:-1]] + [out_vec_lengths[dim]]
            enc = tf.tile(enc, tile_shape)
            dims_enc = enc if dims_enc is None else tf.concat([dims_enc, enc], axis=-1)
        
        return dims_enc
        
    return call

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
        rate = np.pi*tf.cast(vals, dtype=dtype) / scale
        sin = tf.sin(rate)
        cos = tf.cos(rate)
        encoding = tf.concat([sin, cos], axis=-1)
        return encoding
    
    return call

# scale is the max-min of vals
# for mnist it's 28 because thats the width and height of the images
def dual_positional_encoding(image_size, pos_embd_fn):
    n_rows = image_size[0]
    n_cols = image_size[1]
    
    def call(idxs):
        rows = idxs // n_rows
        cols = idxs % n_cols
        
        row_enc = pos_embd_fn(rows)
        col_enc = pos_embd_fn(cols)
        
        encoding = tf.concat([row_enc, col_enc], axis=-1)
        return encoding
        
    return call


def relative_position_matrix(image_size, pos_embd_fn):
    n_rows = image_size[0]
    n_cols = image_size[1]
    
    def call(q_idx, k_idx):
        q_rows = q_idx // n_rows
        k_rows = k_idx // n_rows
        q_cols = q_idx % n_cols
        k_cols = k_idx % n_cols
        
        a_rows = tf.expand_dims(k_rows, -2) - tf.expand_dims(q_rows, -1)
        a_cols = tf.expand_dims(k_cols, -2) - tf.expand_dims(q_cols, -1)
        
        row_enc = pos_embd_fn(a_rows)
        col_enc = pos_embd_fn(a_cols)
        
        encoding = tf.concat([row_enc, col_enc], axis=-1)
        return encoding

    return call


def transformer(m):
    
    colors = Input([None], name="colors")
    inp_idxs = Input([None], name="inp_idxs")
    tar_idxs = Input([None], name="tar_idxs")
    # use type_spec argument because we don't want the implicit batch dim for these inputs
    enc_a_mask = Input(type_spec=tf.TensorSpec(shape=[None, None]), name="enc_mask")
    dec_mask = Input(type_spec=tf.TensorSpec(shape=[None, None]), name="dec_mask")

    tar_shape = tf.shape(tar_idxs)
    # shapes for tf_distribution
    batch_shape = tar_shape[0]
    seq_shape = tar_shape[1]
    event_shape = ()
    
    if m.activation == 'relu':
        m.activation_fn = tf.nn.relu
    elif m.activation == 'swish':
        m.activation_fn = tf.nn.silu
    elif m.activation == 'gelu':
        m.activation_fn = tf.nn.gelu
    
    pos_embd_fn = pos_enc(m.embd_dim//2, scale=100)
    if m.discrete:
        col_embd = layers.Embedding(m.n_colors, m.embd_dim)(colors)
    else:
        col_embd = layers.Dense(m.embd_dim)(colors[:, :, None])
    
    if m.position_embedding == 'pos_enc':
        # set length to somewhat bigger than the image width/height
        position_embedding = dual_positional_encoding(m.image_size, pos_embd_fn)
    elif m.position_embedding == 'learned':
        # learned vector codebook embedding
        position_embedding = layers.Embedding(m.seq_len, m.embd_dim)
    elif m.position_embedding == 'linear':
        # linear mapping from idx to 0..1 range for each position dimension
        position_embedding = linear_position_encoding(m.image_size, [m.embd_dim // 2, m.embd_dim - (m.embd_dim // 2)])
    elif m.position_embedding == 'linear_learned':
        lin_embd = linear_position_encoding(m.image_size, [1, 1])
        position_embedding = lambda x_inp: layers.Dense(m.embd_dim)(layers.Dense(m.embd_dim, activation=m.activation_fn)(lin_embd(x_inp)))
    elif m.position_embedding == 'pos_and_embd':

        frame_embd = pos_enc(m.embd_dim//2, scale=10000)
        dof_embd = layers.Embedding(m.seq_len, m.embd_dim//2)
        def pos_and_embd(idxx):
            frame_idx = idxx // m.n_dof
            dof_idx = idxx % m.n_dof

            return tf.concat([frame_embd(frame_idx), dof_embd(dof_idx)], axis=-1)
    
        position_embedding = pos_and_embd

    inp_pos_embd = position_embedding(inp_idxs)
    tar_pos_embd = position_embedding(tar_idxs)
    
    if m.architecture == 'anp':
        xa = col_embd + inp_pos_embd
        final_hidden = anp_architecture(m)(xa, inp_pos_embd, tar_pos_embd,  enc_a_mask, dec_mask)    
    elif m.architecture == 'deberta_anp':
        make_relative_position_matrix = relative_position_matrix(m.image_size, pos_embd_fn)
        xa = col_embd
        if m.use_relative_positions:
            r_self = make_relative_position_matrix(q_idx=inp_idxs, k_idx=inp_idxs)
            r_cross = make_relative_position_matrix(q_idx=tar_idxs, k_idx=inp_idxs)
        else:
            r_self = r_cross = None
        final_hidden = deberta_anp_architecture(m)(xa, inp_pos_embd, tar_pos_embd, r_self, r_cross, enc_a_mask, dec_mask)
    else:
        raise f"invalid architecture '{m.architecture}'"
    


    if m.continuous:
        loc = layers.Dense(1, activation=None, name='loc')(final_hidden) # loc
        concentration = layers.Dense(1, activation='relu', name='concentration')(final_hidden) # conentration

        outputs = [
            layers.concatenate([loc, concentration])
        ]

    else:
        outputs = [
            tf.keras.layers.Dense(m.n_colors)(final_hidden)
        ]
    
    return Model(inputs=[colors, inp_idxs, tar_idxs, enc_a_mask, dec_mask], outputs=outputs)
    
