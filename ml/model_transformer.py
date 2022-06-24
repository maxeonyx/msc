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
                decay_rate=0.96
            )


    def __call__(self, step):
        return tf.cond(
            pred=tf.less(step, self.warmup_steps),
            true_fn=lambda: self.peak_learning_rate * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32),
            false_fn=lambda: self.other_schedule(step)
        )

class FeedforwardWrapper(Model):

    def predict(self, x, y, n_frames):
        x = x.copy()

        batch_size = tf.shape(x["angles"])[0]

        output = self(x)
        angles = self.prediction_head.sample(output)
        if not self.prediction_head.sample_is_mean:
            pred_angles_mean = self.prediction_head.mean(output)
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

                    output = self({
                        "angles": angles,
                        "frame_idxs": frame_idxs,
                        "hand_idxs": hand_idxs,
                        "dof_idxs": dof_idxs,
                    })[..., -1:, :] # transformer outputs a sequence, but we only need the new token

                    new_angles = self.prediction_head.sample(output)
                    if not self.prediction_head.sample_is_mean:
                        new_pred_angles_mean = self.prediction_head.mean(output)

                    angles = tf.concat([angles, new_angles], axis=-1)
                    if not self.prediction_head.sample_is_mean:
                        pred_angles_mean = tf.concat([pred_angles_mean, new_pred_angles_mean], axis=-1)

        if not self.prediction_head.sample_is_mean:
            return angles[..., :-1], pred_angles_mean[..., :-1]
        return angles[..., :-1], None # remove last output because otherwise we have n+1 which doesn't evenly divide



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
        super(TransformerBlock, self).__init__()
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


class KerasTransformer(FeedforwardWrapper):
    def __init__(self, cfg, embedder, prediction_head, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cfg = cfg | cfg.transformer
        self.cfg = cfg
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
