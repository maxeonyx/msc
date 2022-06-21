import math
from tracemalloc import start

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, Input

from ml import data_tf

class WarmupLRSchedule(keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, peak_learning_rate, warmup_steps, other_schedule=None):
        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps
        self.other_schedule = other_schedule
        if other_schedule is None:
            self.other_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=peak_learning_rate,
                decay_steps=1000,
                decay_rate=0.99
            )


    def __call__(self, step):
        return tf.cond(
            pred=tf.less(step, self.warmup_steps),
            true_fn=lambda: self.peak_learning_rate * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32),
            false_fn=lambda: self.other_schedule(step)
        )
    


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

def multi_head_attention(cfg):
    embd_dim, n_heads = cfg.embd_dim, cfg.n_heads
    
    wk = layers.Dense(embd_dim)
    wq = layers.Dense(embd_dim)
    wv = layers.Dense(embd_dim)
    
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

def pointwise_feedforward_layer(cfg, output_layer, n_hidden_layers=1, dtype=None):
    hidden_layers = [layers.Dense(cfg.ffl_dim) for _ in range(n_hidden_layers)]
    dtype = dtype or tf.keras.mixed_precision.global_policy()

    
    if cfg.activation == 'relu':
        activation_fn = tf.nn.relu
    elif cfg.activation == 'swish':
        activation_fn = tf.nn.silu
    elif cfg.activation == 'gelu':
        activation_fn = tf.nn.gelu
    else:
        raise ValueError('Unknown activation function: {}'.format(cfg.activation))
    
    def call(x):
        for layer in hidden_layers:
            x = layer(x)
            x = activation_fn(x)
        if output_layer is not None:
            x = output_layer(x)
        return x
    return call

def transformer_layer(cfg):
    mha = multi_head_attention(cfg)
    ffl = pointwise_feedforward_layer(cfg, output_layer=layers.Dense(cfg.embd_dim))
    layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    dropout1 = layers.Dropout(cfg.dropout_rate)
    dropout2 = layers.Dropout(cfg.dropout_rate)
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

# gpt style transformer
def gpt(cfg):
    enc_layers = [transformer_layer(cfg) for _ in range(cfg.n_enc_layers)]
    def call(embd, enc_mask):
        for enc_layer in enc_layers:
            embd += enc_layer(embd, embd, mask=enc_mask)
        return embd
    return call

class Transformer(Model):
    def __init__(self, cfg, embedder, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cfg = cfg
        self.embedder = embedder
        self.transformer = gpt(cfg | cfg.transformer)
    
    def call(self, inputs):
        embd = self.embedder.embed_sequence_with_begin_sentinel(inputs)
        seq_len = tf.shape(embd)[1]
        enc_mask = create_look_backward_equal_mask(seq_len, seq_len)
        embd = self.transformer(embd, enc_mask)
        angles = self.embedder.unembed(embd)
        return angles

    def predict(self, x, y, n_frames):
        x = x.copy()

        batch_size = tf.shape(x["angles"])[0]

        angles = self(x)
        frame_idxs = x["frame_idxs"]
        hand_idxs = x["hand_idxs"]
        dof_idxs = x["dof_idxs"]
        
        start_frame = frame_idxs[..., -1] + 1
        for i_frame in tf.range(start_frame, start_frame + n_frames):
            for i_hand in tf.range(self.cfg.n_hands):
                for i_dof in tf.range(self.cfg.n_dof):
                    # tile a constant value to the batch dimension and len=1 seq dim
                    tile_batch_seq = lambda x: tf.tile(x[None, None], [batch_size, 1])
                    
                    if self.cfg.relative_frame_idxs and not self.cfg.target_is_sequence:
                        tile_batch = lambda x: tf.tile(x[None, :], [batch_size, 1])
                        frame_idxs = tile_batch(data_tf.frame_idxs_for(self.cfg, i_hand * self.cfg.n_dof + i_dof, tf.shape(angles)[-1]))
                    elif self.cfg.relative_frame_idxs:
                        raise NotImplementedError("Transformer does not yet support relative frame indices")
                    else:
                        frame_idxs = tf.concat([frame_idxs, tile_batch_seq(i_frame)], axis=1)
                    hand_idxs = tf.concat([hand_idxs, tile_batch_seq(i_hand)], axis=-1)
                    dof_idxs  = tf.concat([dof_idxs,  tile_batch_seq(i_dof)],  axis=-1)

                    new_angles = self({
                        "angles": angles,
                        "frame_idxs": frame_idxs,
                        "hand_idxs": hand_idxs,
                        "dof_idxs": dof_idxs,
                    })[..., -1:] # transformer outputs a sequence, but we only need the new token

                    angles = tf.concat([angles, new_angles], axis=-1)

        return angles[..., :-1] # remove last output because otherwise we have n+1 which doesn't evenly divide
