import einops as ein
from einops.layers.keras import EinMix
import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Union, List

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
        Size of 
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


class RelativePositionProjection(tf.keras.Layer):

    def __init__(self, ein_expr, weight_shape, target_N, max_dist, D, QK, rpe):
        self.target_N = target_N
        self.max_dist = max_dist
        self.rpe = rpe
        # seq dim should be first axis for the tf.gather below
        self.w = EinMix(ein_expr, weight_shape=weight_shape, d=D, qk=QK)
        self.projections = None
    
    def call(self, relative_pos):
        if self.projections is None:
            rel_embd = self.rpe.get_all_embeddings()
            self.projections = self.w(rel_embd)
        indices = self.rpe.get_indices(relative_pos)
        rel_proj = tf.gather(self.projections, indices, axis=0)
        return rel_proj


class IncrementalRelativeMultiQueryAttention(tf.Module):

    def __init__(self, max_N, QK, V, D, H, rpe):
        super().__init__()

        self.QK = QK
        self.V = V
        self.D = D
        self.H = H

        self.wv = EinMix('b m d -> m b v', weight_shape='d v', d=self.D, v=self.V)

        self.wqc = EinMix('b m d -> b h m qk', weight_shape='d h qk', d=self.D, h=self.H, qk=self.QK)
        # k & v have m dimension first so they can be put into TensorArray
        self.wkc = EinMix('b m d -> m b qk', weight_shape='d qk', d=self.D, qk=self.QK)

        self.wqp = RelativePositionProjection(
            'b m d -> b h m n qk', # TODO: This is BIIIIG....
            weight_shape='d h qk',
            max_N=max_N,
            D=D,
            QK=QK,
            rpe=rpe
        )
        self.wkp = RelativePositionProjection(
            'b m d -> b m n qk',
            weight_shape='d qk',
            max_N=max_N,
            D=D,
            QK=QK,
            rpe=rpe
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


    def call_incremental(self, zq_embd, zq_rel_embd, mask, training):

        # M is the number of new inputs this iteration. Might be M=N, or might be M=1.
        # N is the total number of inputs so far, including the current input. (so N >= M)

        i = self.k_cache.size()

        v = self.wv(zq_embd)
        self.v_cache.write(i, v)
        v = self.v_cache.concat()

        n_attn_types = 3
        scale = 1. / tf.sqrt(self.D * n_attn_types)

        # c2c
        # c
        qc = self.wqc(zq_embd)
        # c
        kc = self.wkc(zq_embd)

        self.kc_cache.write(i, kc)
        kc = self.kc_cache.concat()

        c2c_attn_logits = tf.einsum('nbk,bhmk->bhmn', kc, qc)

        # p2c
        # p
        kp = self.wqp(zq_rel_embd)

        # c
        c2c_attn_logits = tf.einsum('bmnk,bhmk->bhmn', kp, qc)


        # p
        kp = self.wkp()
        vp = self.wvp(zq_embd)





        c2c_attn_probs = self.softmax(c2c_attn_logits, mask)
        






def embedder(x):

    angles = x["angles"]

def incremental_relative_multiquery_attention():
