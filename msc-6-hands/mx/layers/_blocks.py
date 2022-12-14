from mx.prelude import *
from mx.utils import Einshape

def featurewise_dense(in_dims: Einshape, out_dims: Einshape, regularizer=None, name="mix") -> Model:
    """
    A dense layer across the feature dimensions.
    """

    in_rearrange = lambda t: ein.rearrange(t, f"... {in_dims.f_str} -> ... ({in_dims.f_str})", **in_dims.f_dict)
    out_rearrange = lambda t: ein.rearrange(t, f"... ({out_dims.f_str}) -> ... {out_dims.f_str}", **out_dims.f_dict)

    dense = layers.Dense(
        out_dims.f_product,
        use_bias=False,
        name=name,
        kernel_regularizer=regularizer,
    )

    def call(embd):

        embd = in_rearrange(embd)
        embd = dense(embd)
        embd = out_rearrange(embd)

        return embd

    inputs = u.input_dict(
        Input(shape=in_dims.s_f_shape, name="embd"),
    )

    return Model(inputs=inputs, outputs=call(**inputs), name=name)

def featurewise_dense_block(hidden_size: int, in_dims: Einshape, out_dims: Einshape, regularizer=None, name="mix") -> Model:
    """
    Two dense layers with an activation in between, applied across all feature dimensions.
    """

    hidden_dims = in_dims.with_feature_dims({ "hidden": hidden_size })

    in_layer = featurewise_dense(in_dims, hidden_dims, regularizer=regularizer, name=f"{name}/in")
    activation = layers.ReLU(name=f"{name}/act")
    out_layer = featurewise_dense(hidden_dims, out_dims, regularizer=regularizer, name=f"{name}/out")

    def call(embd):

        embd = in_layer(embd)
        embd = activation(embd)
        embd = out_layer(embd)

        return embd

    inputs = u.input_dict(
        Input(shape=in_dims.s_f_shape, name="embd"),
    )

    return Model(inputs=inputs, outputs=call(**inputs), name=name)


def mlp(embd_dim, hidden_units, dropout=0.1, name="mlp") -> Model:
    """
    MLP with dropout
    """

    def call(embd):
        embd = layers.Dense(hidden_units, activation="relu")(embd)
        embd = layers.Dropout(dropout)(embd)
        return embd

    inputs = u.input_dict(
        Input(shape=[embd_dim], name="embd"),
    )

    return Model(inputs=inputs, outputs=call(**inputs), name=name)


def mha(embd_shape: Einshape, n_heads: int, kv_embd_shape: Einshape = None, normalization_type: Literal["scale", "softmax"] = "softmax", type: Literal["self_attn", "cross_attn"] = "self_attn", rel_idxs: Union[Literal[None, True], int] = None, causal_mask=True, return_attn_weights=False, name="scale_mha") -> Model:

    q_proj = layers.Dense(embd_shape.f_product, use_bias=False, name=f"{name}/q_proj")
    k_proj = layers.Dense(embd_shape.f_product, use_bias=False, name=f"{name}/k_proj")
    v_proj = layers.Dense(embd_shape.f_product, use_bias=False, name=f"{name}/v_proj")

    rearrange_embd_in = lambda t: ein.rearrange(t, f"... {embd_shape.s_str} {embd_shape.f_str} -> ... ({embd_shape.s_str}) ({embd_shape.f_str})", **embd_shape.f_dict, **embd_shape.s_dict)
    rearrange_embd_out = lambda t: ein.rearrange(t, f"... ({embd_shape.s_str}) ({embd_shape.f_str}) -> ... {embd_shape.s_str} {embd_shape.f_str}", **embd_shape.f_dict, **embd_shape.s_dict)
    if kv_embd_shape is not None:
        rearrange_kv_embd_in = lambda t: ein.rearrange(t, f"... {kv_embd_shape.s_str} {kv_embd_shape.f_str} -> ... ({kv_embd_shape.s_str}) ({kv_embd_shape.f_str})", **kv_embd_shape.f_dict, **kv_embd_shape.s_dict)
        rearrange_kv_embd_out = lambda t: ein.rearrange(t, f"... ({kv_embd_shape.s_str}) ({kv_embd_shape.f_str}) -> ... {kv_embd_shape.s_str} {kv_embd_shape.f_str}", **kv_embd_shape.f_dict, **kv_embd_shape.s_dict)

    def call(inputs):

        if type == "self_attn":
            embd = rearrange_embd_in(inputs["embd"])
            q, k, v = q_proj(embd), k_proj(embd), v_proj(embd)
        else:
            q_embd = rearrange_embd_in(inputs["q_embd"])
            kv_embd = rearrange_kv_embd_in(inputs["kv_embd"])
            q, k, v = q_proj(q_embd), k_proj(kv_embd), v_proj(kv_embd)

        q = ein.rearrange(q, "b m (h d) -> b h m d", h=n_heads)
        k = ein.rearrange(k, "b n (h d) -> b h n d", h=n_heads)
        v = ein.rearrange(v, "b n (h d) -> b h n d", h=n_heads)

        attn_logits = tf.einsum("b h m d, b h n d -> b h m n", q, k)

        if causal_mask:
            if type == "self_attn":
                mask, mask_scales = u.make_causal_mask(embd_shape.s_product)
            else: # cross_attn
                mask, mask_scales = u.make_causal_mask(embd_shape.s_product, kv_embd_shape.s_product)

        if normalization_type == "softmax":

            # scale to unit vectors
            e = tf.cast(embd_shape.f_product, u.dtype())
            attn_logits = attn_logits * 1./tf.sqrt(e)

            if causal_mask:
                attn_logits -= mask * 1e9
            attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        elif normalization_type == "scale":
            # we assume that the query and key vectors are already scaled by 1/sqrt(embd_dim)
            # and that their 2-norm is approximately 1. Therefore, their
            # dot product is approximately the cosine similarity between the two vectors.
            # which is in the range [-1, 1].
            if causal_mask:
                attn_logits *= mask
                scales = mask_scales
            else:
                d = tf.cast(embd_shape.s_product, u.dtype())
                scales = 1./tf.sqrt(d)
                scales = scales[None]

            # scale the attention weights according to the number of value vectors that
            # are being combined to produce the output vector. When we use the mask,
            # this is the number of non-masked-out values.
            attn_weights = attn_logits * scales
        else:
            raise ValueError(f"Unknown weight_type '{normalization_type}' for mha")

        out = tf.einsum("b h m n, b h n d -> b h m d", attn_weights, v)
        out = ein.rearrange(out, "b h m d -> b m (h d)")

        if return_attn_weights:
            return out, attn_weights

        return out

    if type == "self_attn":
        embd_inputs = [
            Input(shape=embd_shape.s_f_shape, name="embd"),
        ]
    elif type == "cross_attn":
        assert kv_embd_shape is not None, "kv_embd_shape must be specified for cross attention"
        embd_inputs = [
            Input(shape=embd_shape.s_f_shape, name="q_embd"),
            Input(shape=kv_embd_shape.s_f_shape, name="kv_embd"),
        ]
    else:
        raise ValueError(f"Unknown type '{type}' for mha")

    # if rel_idxs is None:
    #     idx_inputs = []
    # else:
    #     if rel_idxs is True:
    #         idxs_shape = [seq_dim]
    #     elif type(rel_idxs) is int:
    #         idxs_shape = [seq_dim]
    #     else:
    #         raise ValueError("rel_idxs must be None, True or int. Got: ", rel_idxs)

    #     if self_attn:
    #         idx_inputs = [
    #             Input(shape=idxs_shape, name="qk_idxs"),
    #         ]
    #     else:
    #         idx_inputs = [
    #             Input(shape=idxs_shape, name="q_idxs"),
    #             Input(shape=idxs_shape, name="k_idxs"),
    #         ]


    inputs = u.input_dict(
        *embd_inputs,
        # *idx_inputs,
    )

    return Model(inputs=inputs, outputs=call(inputs), name=name)
