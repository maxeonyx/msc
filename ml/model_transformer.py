import math
from tracemalloc import start

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, Input

from ml import data_tf, models

class WarmupLRSchedule(keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, peak_learning_rate, warmup_steps, other_schedule=None):
        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps
        self.other_schedule = other_schedule
        if other_schedule is None:
            self.other_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=peak_learning_rate,
                decay_steps=1000,
                decay_rate=0.96
            )

    def __call__(self, step):
        return tf.cond(
            pred=tf.less(step, self.warmup_steps),
            true_fn=lambda: self.peak_learning_rate * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32),
            false_fn=lambda: self.other_schedule(step)
        )
    
    def get_config(self):
        return {
            'peak_learning_rate': self.peak_learning_rate,
            'warmup_steps': self.warmup_steps,
            'other_schedule': self.other_schedule
        }




############ keras transformer impl

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


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class KerasTransformer(models.SequenceModelBase):
    def __init__(self, cfg, embedder, prediction_head, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        cfg = cfg | cfg.transformer
        self.embedder = embedder
        self.prediction_head = prediction_head
        self.transformer_layers = [
            TransformerBlock(cfg.embd_dim, cfg.n_heads, cfg.ffl_dim, cfg.dropout_rate)
            for _ in range(cfg.n_layers)
        ]
    
    def call(self, inputs):
        embd = self.embedder.embed_sequence_with_begin_sentinel(inputs, length=1)
        for layer in self.transformer_layers:
            embd = layer(embd)
        return self.prediction_head.unembed(embd)
