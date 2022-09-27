from types import FunctionType
import einops as ein
import tensorflow as tf
import numpy as np
import functools
from math import pi, tau
from typing import Callable, Collection, Literal, Tuple, TypedDict, Union, List
import typing
if typing.TYPE_CHECKING:
    import keras.api._v2.keras as keras
    from keras.api._v2.keras import Model, Input, layers
else:
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers
from tensorflow_probability import distributions as tfd

from ._layer_utils import input_dict, make_causal_mask, shape_list
from mx.utils import Einshape

def featurewise_dense(in_dims: Einshape, out_dims: Einshape, regularizer, name="mix") -> Model:
    """
    A dense layer across the feature dimensions.
    """

    in_dims = [shape for _, shape in in_dims]
    out_dims = [shape for _, shape in out_dims]

    in_rearrange = f"... {in_dims.f_str} -> ... ({in_dims.f_str})"
    out_rearrange = f"... ({out_dims.f_str}) -> ... {out_dims.f_str}"

    dense = layers.Dense(
        out_dims.f_product,
        use_bias=False,
        name=name,
        kernel_regularizer=regularizer,
    )

    def call(embd):

        embd = ein.rearrange(embd, in_rearrange, **in_dims.f)

        embd = dense(embd)

        embd = ein.rearrange(embd, out_rearrange, **out_dims.f)

        return embd

    inputs = input_dict(
        Input(shape=in_dims.s_f_shape, name="embd"),
    )
    
    return Model(inputs=inputs, outputs=call(inputs), name=name)


def mlp(embd_dim, hidden_units, dropout=0.1, name="mlp") -> Model:
    """
    MLP with dropout
    """

    def call(embd):
        embd = layers.Dense(hidden_units, activation="relu")(embd)
        embd = layers.Dropout(dropout)(embd)
        return embd

    inputs = input_dict(
        Input(shape=[embd_dim], name="embd"),
    )
    
    return Model(inputs=inputs, outputs=call(**inputs), name=name)


def mha(n_heads=8, weight_type: Literal["scale", "softmax"] = "softmax", seq_dim: int = None, embd_dim:int=None, self_attn=True, rel_idxs:None|True|int=None, causal_mask=True, return_attn_weights=False, name="scale_mha") -> Model:

    q_proj = layers.Dense(embd_dim, use_bias=False, name=f"{name}/q_proj")
    k_proj = layers.Dense(embd_dim, use_bias=False, name=f"{name}/k_proj")
    v_proj = layers.Dense(embd_dim, use_bias=False, name=f"{name}/v_proj")

    def call(**inputs):

        if self_attn:
            q = inputs["qk"]
            k = inputs["qk"]
        else:
            q = inputs["q"]
            k = inputs["k"]
        
        v = inputs["v"]

        q = q_proj(q)
        k = k_proj(k)
        v = v_proj(v)

        q = ein.rearrange(q, "b m (h d) -> b h m d", h=n_heads)
        k = ein.rearrange(k, "b n (h d) -> b h n d", h=n_heads)
        v = ein.repeat(v, "b n d -> b h n d", h=n_heads)

        # we assume that the query and key vectors are already scaled by 1/sqrt(d)
        # and that their 2-norm is approximately 1. Therefore, their
        # dot product is approximately the cosine similarity between the two vectors.
        # which is in the range [-1, 1].
        attn_logits = tf.einsum("b h m d, b h n d -> b h m n", q, k)

        if weight_type == "softmax":
            if causal_mask:
                mask, _scales = make_causal_mask(seq_dim)
                attn_logits += mask * -1e9
            attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        elif weight_type == "scale":
            if causal_mask:
                mask, scales = make_causal_mask(seq_dim)
                attn_logits *= mask
            else:
                scales = tf.expand_dims(1./tf.sqrt(tf.cast(seq_dim, tf.float32)), 0)
            
            # scale the attention weights according to the number of value vectors that
            # are being combined to produce the output vector. When we use the mask,
            # this is the number of non-masked-out values.
            attn_weights = attn_logits * scales
        else:
            raise ValueError(f"Unknown weight_type '{weight_type}' for mha")
        
        attn = tf.einsum("b h m n, b h n d -> b h m d", attn_weights, v)

        if return_attn_weights:
            return attn, attn_weights
        
        return attn
    
    if self_attn:
        embd_inputs = [
            Input(shape=(seq_dim, embd_dim), name="qk"),
            Input(shape=(seq_dim, embd_dim), name="v"),
        ]
    else:
        embd_inputs = [
            Input(shape=(seq_dim, embd_dim), name="q"),
            Input(shape=(seq_dim, embd_dim), name="k"),
            Input(shape=(seq_dim, embd_dim), name="v"),
        ]

    if rel_idxs is None:
        idx_inputs = []
    else:
        if rel_idxs is True:
            idxs_shape = [seq_dim]
        elif type(rel_idxs) is int:
            idxs_shape = [seq_dim]
        else:
            raise ValueError("rel_idxs must be None, True or int. Got: ", rel_idxs)
        
        if self_attn:
            idx_inputs = [
                Input(shape=idxs_shape, name="qk_idxs"),
            ]
        else:
            idx_inputs = [
                Input(shape=idxs_shape, name="q_idxs"),
                Input(shape=idxs_shape, name="k_idxs"),
            ]


    inputs = input_dict(
        *embd_inputs,
        *idx_inputs,
    )

    return Model(inputs=inputs, outputs=call(**inputs), name=name)
