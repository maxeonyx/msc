import einops as ein
from einops.layers.keras import EinMix
import tensorflow as tf
import typing
if typing.TYPE_CHECKING:
    from tensorflow.python import keras
    from tensorflow.python.keras import Input, Model, layers
else:
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers
import numpy as np
from typing import Any, Optional, Tuple, Union, List

import deberta

def random_foveal_context(cfg, n_frames, x):

    shape = tf.shape(x["angles"])
    n_total_frames = shape[0]
    
    x["angles"] = rearrange(x["angles"], 'b, f, h, d -> b, f, (h, d)')
    x["frame_idxs"] = rearrange(x["frame_idxs"], 'b, f, h, d -> b, f, (h, d)')
    x["hand_idxs"] = rearrange(x["hand_idxs"], 'b, f, h, d -> b, f, (h, d)')
    x["dof_idxs"] = rearrange(x["dof_idxs"], 'b, f, h, d -> b, f, (h, d)')

    tok_per_frame = cfg.n_hands * cfg.n_dof
    chunk_size = size

    if aligned:
        # get a random index
        idx = tf.random.uniform([], minval=0, maxval=n_frames-chunk_size, dtype=tf.int32)
        idx = idx * tok_per_frame
    else:
        idx = tf.random.uniform([], minval=0, maxval=n_frames*tok_per_frame-chunk_toks, dtype=tf.int32)

    x["angles"] = x["angles"][idx:idx+chunk_toks]

    if cfg.relative_frame_idxs:
        x["frame_idxs"] = frame_idxs_for(cfg, idx % tok_per_frame, chunk_toks)
    else:
        x["frame_idxs"] = x["frame_idxs"][idx:idx+chunk_toks]


    x["hand_idxs"] = x["hand_idxs"][idx:idx+chunk_toks]
    x["dof_idxs"] = x["dof_idxs"][idx:idx+chunk_toks]

    return x


def bucket_scale_fn(decay_power, step_power, clip=1000):
    """
    
    At x = 
    
    """

    ln = tf.math.log
    pow = tf.math.pow

    def g(x):
        ((x+1) * ln(step_power) * pow(decay_power, -(ln(x+1) / ln(10))  ))   /   ln(decay_power/step_power)

    def center(f):
        return lambda x: f(x) - f(0)

    def make_symmetric(f):
        return lambda x: tf.math.sign(x) * f(tf.math.abs(x))

    def do_clip(f):
        return lambda x: x if clip is None else tf.clip_by_value(f(x), -clip, clip)

    return do_clip(make_symmetric(center(g)))


class RelativePositionEmbedding(tf.keras.Layer):

    def __init__(self, max_N, D,  allow_negative=False):
        """
        If the computation only ever uses
        relative attention in one direction (eg. incremental multi-query layers with causal masking),
        then allow_negative can be set to false to halve memory use.
        """
        self.max_N = max_N
        if allow_negative:
            self.embedding = tf.keras.layers.Embedding(self.max_N * 2 - 1, D)
        else:
            self.embedding = tf.keras.layers.Embedding(self.max_N, D)
        self.allow_negative = allow_negative

        # every power of ten, double the width of the buckets.
        # starts at 1 bucket per 1 distance
        # reaches 1 bucket per 2 distance at relative distance=10
        # reaches 1 bucket per 4 distance at relative distance=100
        self.bucket_scale_fn = bucket_scale_fn(decay_power=2., step_power=10., clip=(max_N-1))
    
    def get_indices(self, relative_pos):
        if self.allow_negative:
            indices = self.bucket_scale_fn(relative_pos) + self.max_N - 1
        else:
            indices = self.bucket_scale_fn(relative_pos)
        return indices
    
    def get_all_embeddings(self):
        if self.allow_negative:
            indices = tf.range(self.max_N * 2 - 1)
        else:
            indices = tf.range(self.max_N)
        return self.get_embedding(indices)

    # log-bucketing of relative position embeddings
    # bucketing are small near 0, and grow larger with distance.
    # at x = 0, there is 1 bucket (index) per 1 distance
    # at x = 10, there is 1 bucket (index) per e distance
    # at x = 100, there is 1 bucket (index) per 2e distance
    def get_embedding(self, indices):
        # indices are dense near the center, 
        # ceil leaves a bucket at 0
        return self.embedding(indices)


class RelativePositionProjectionLookup(tf.keras.Layer):

    def __init__(self, projection, relative_pos_embedding):
        self.rpe = relative_pos_embedding
        # seq dim should be first axis for the tf.gather below
        self.projection = projection
        self.cache = None
    
    def call(self, relative_pos):
        if self.cache is None:
            rel_embd = self.rpe.get_all_embeddings()
            self.cache = self.projection(rel_embd)
        indices = self.rpe.get_indices(relative_pos)
        rel_proj = tf.gather(self.cache, indices, axis=0)
        return rel_proj


class IncrementalRelativeMultiQueryAttention(tf.Module):

    def __init__(self, QK, V, D, H, rpe):
        super().__init__()

        self.QK = QK
        self.V = V
        self.D = D
        self.H = H

        self.wv = EinMix('b m d -> m b v', weight_shape='d v', d=self.D, v=self.V)

        self.wqc = EinMix('b m d -> b h m qk', weight_shape='d h qk', d=self.D, h=self.H, qk=self.QK)
        # k & v have m dimension first so they can be put into TensorArray
        self.wkc = EinMix('b m d -> m b qk', weight_shape='d qk', d=self.D, qk=self.QK)

        self.wqp = RelativePositionProjectionLookup(
            # TODO: This is BIIIIG....
            projection=EinMix(
                # ... is max_N
                '... d -> ... h qk',
                weight_shape='d h qk',
                H=H,
                D=D,
                QK=QK,
            ),
            rpe=rpe,
        )
        self.wkp = RelativePositionProjectionLookup(
            projection=EinMix(
                'n d -> n qk',
                weight_shape='d qk',
                D=D,
                QK=QK,
            ),
            rpe=rpe,
        )

        self.wo = EinMix('b m h v -> b m d', weight_shape='h v d', v=self.V, h=self.H, d=self.D)

        self.softmax = deberta.TFDebertaXSoftmax(axis=-1)
        self.dropout = deberta.TFDebertaStableDropout(0.1)
        self.pos_dropout = deberta.TFDebertaStableDropout(0.1)

    def reset_cache(self, batch_size, max_N):

        self.wqp.reset_cache()
        self.wkp.reset_cache()

        # initialize TensorArrays with size N
        self.kc_cache = tf.TensorArray(
            element_shape=[None, batch_size, self.QK],
            size=max_N,
            dynamic_size=True,
            clear_after_read=False,
            dtype=tf.float32,
        )
        self.v_cache = tf.TensorArray(
            element_shape=[None, batch_size, self.V],
            size=max_N,
            dynamic_size=True,
            clear_after_read=False,
            dtype=tf.float32,
        )


    def call_incremental(self, embd, rel_pos, mask, training):

        # M is the number of new inputs this iteration. Might be M=N, or might be M=1.
        # N is the total number of inputs so far, including the current input. (so N >= M)

        # embd is [B, M, D]
        # rel_pos is [B, M, N] (int32)
        # mask is [N, N]
        # returns [B, M, D]
        # and increases N_t to N_{t+1}= N_t + M

        i = self.k_cache.size()

        v = self.wv(embd)
        self.v_cache.write(i, v)
        v = self.v_cache.concat()

        n_attn_types = 3
        scale = 1. / tf.sqrt(self.D * n_attn_types)

        # c2c
        qc = self.wqc(embd)
        kc = self.wkc(embd)

        self.kc_cache.write(i, kc)
        kc = self.kc_cache.concat()

        c2c_attn_logits = tf.einsum('bhmk,bnk->bhmn', qc, kc)

        # p2c
        qp = self.wqp(rel_pos)
        # kc
        # h second last in this one because of embedding lookup
        p2c_attn_logits = tf.einsum('bmnhk,bnk->bhmn', qp, kc)

        # c2p
        # qc
        kp = self.wkp(rel_pos)
        c2p_attn_logits = tf.einsum('bhmk,bmnk->bhmn', qc, kp)

        
        attn_logits = (c2c_attn_logits + p2c_attn_logits + c2p_attn_logits) * scale
        attn_logits = self.dropout(attn_logits, training=training)
        attn_weights = self.softmax(attn_logits, mask)
    
        v_h = tf.einsum('bhmn,bnv->bhmv', attn_weights, v) # value-per-head
        v_o = self.wo(v_h) # project to output dimension
        return v_o



class IRMQAT(tf.keras.layers.Layer):

    def __init__(self, cfg, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_layers = cfg.n_layers
        self.embd_dim = cfg.embd_dim
        
        self.embd_vecs = EinMix(
            pattern='f h d s -> f e',
            weight_shape='h d s e',
            e=cfg.embd_dim,
            h=cfg.n_hands,
            d=cfg.n_dof,
            s=2
        )

        self.rpe = RelativePositionEmbedding(
            # maximum number of inputs.
            # defines the size of the single "rel_embd" cache
            # and n_layers "rel_proj" caches
            max_N=cfg.max_N,
            D=cfg.embd_dim,
        )
        
        self.irmqa_layers = [
            [
                IncrementalRelativeMultiQueryAttention(
                    QK=cfg.qk_dim,
                    V=cfg.v_dim,
                    D=cfg.d_dim,
                    H=cfg.n_heads,
                    rpe=self.rpe,
                ),
                deberta.TFDebertaIntermediate(
                    intermediate_size=cfg.intermediate_size,
                    initializer_range=cfg.initializer_range,
                    hidden_act=cfg.hidden_act,
                ),
                deberta.TFDebertaOutput(
                    initializer_range=cfg.initializer_range,
                    hidden_size=cfg.hidden_size,
                    hidden_dropout_prob=cfg.hidden_dropout_prob,
                    layer_norm_eps=cfg.layer_norm_eps,
                )
            ]
        ]
    
    def call(self, embd, rel_pos, mask, training):

        for mqattn, interm, output in self.irmqa_layers:

            embd = mqattn(embd, rel_pos, mask, training)
            embd = interm(embd)
            embd = output(embd)
        
        return embd


def log_weights(n_max, n_samples, decay_by_d_every_power_of_b=(0.5, 10),):
    
    D, B = decay_by_d_every_power_of_b

    normalize = tf.linalg.normalize
    logb1p = lambda b, x: tf.math.log1p(x)/tf.math.log(b)
    powf = lambda b, x: tf.math.pow(tf.cast(b, tf.float32), x)
    
    weights = normalize(powf(D, logb1p(B, tf.range(n))))

def embedder(cfg, x):

    angles = Input(shape=[None, cfg.n_hands, cfg.n_dofs], name='angles')
    angles = tf.ensure_shape(angles, [None, cfg.n_hands, cfg.n_dof])

    n_frames = tf.shape(angles)[0]
    n_hands = tf.shape(angles)[1]
    n_dof = tf.shape(angles)[2]

    sincos = ein.rearrange([tf.sin(angles), tf.cos(angles)], 's f h d -> f h d s')
    
    vec_idxs = tf.random.log_uniform_candidate_sampler(tf.range(n_frames), cfg.max_vec_inputs, )
    n_hand_inputs = tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32)
    n_dof_inputs = tf.random.uniform([], minval=0, maxval=23, dtype=tf.int32)

    vec_inputs = tf.random.
