import tensorflow as tf
from tensorflow.keras import layers, Model, Input
 
def conv(cfg, name="conv"):
    inputs = Input(shape=(None, cfg.embd_dim))

    kernel_size = cfg.width_frames*cfg.n_hands*cfg.n_dof

    embd = inputs
    for _ in range(cfg.n_layers):
        embd = layers.Conv1D(filters=cfg.filters, kernel_size=kernel_size, activation=cfg.activation, padding='causal')(embd)
    
    return Model(inputs=inputs, outputs=embd, name=name)


def mlp(cfg, name="mlp"):
    inputs = Input(shape=(None, cfg.embd_dim))

    embd = inputs
    for _ in range(cfg.n_layers):
        embd = layers.Dense(cfg.hidden_dim, activation=cfg.activation)(embd)
        embd = layers.Dropout(cfg.dropout_rate)(embd)
    embd = layers.Dense(cfg.embd_dim)(embd)
    
    outputs = embd

    return Model(inputs, outputs, name=name)

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)

def transformer_block(cfg, name="transformer_block"):
    inputs = Input(shape=(None, cfg.embd_dim))

    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    seq_len = input_shape[1]
    causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
    attention_output = layers.MultiHeadAttention(cfg.n_heads, cfg.embd_dim)(inputs, inputs, attention_mask=causal_mask)
    attention_output = layers.Dropout(cfg.dropout_rate)(attention_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    ffn_output = layers.Dense(cfg.ffl_dim, activation=cfg.activation)(out1)
    ffn_output = layers.Dense(cfg.embd_dim)(ffn_output)
    ffn_output = layers.Dropout(cfg.dropout_rate)(ffn_output)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return Model(inputs, out2, name=name)

def transformer(cfg, name="transformer"):
    inputs = Input(shape=(None, cfg.embd_dim))

    embd = inputs
    for i in range(cfg.n_layers):
        embd = transformer_block(cfg, name=f"{name}_block_{i}")(embd)
    
    return Model(inputs, embd, name=name)


# def deberta(cfg, name="deberta"):

#     from transformers import models.deberta.TFDebertaEncoder

#     inputs = Input(shape=(None, cfg.embd_dim))

#     encoder = TFDebertaEncoder(
        
#     )
    
#     return Model(inputs, embd, name=name)
