import abc
from functools import reduce
import math
from pprint import pprint
from re import S
import typing
import einops as ein
import tensorflow as tf
from ml import data_tf, encoders, predict, prediction_heads, utils, deberta
from ml.data_bvh import chunk
if typing.TYPE_CHECKING:
    from tensorflow.python import keras
    from tensorflow.python.keras import Input, Model, layers
else:
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers
from tensorflow_probability import distributions as tfd
from ml.utils import tf_scope
import enlighten
from math import pi, tau

@tf_scope
def bucket_scale_fn(decay_power, step_power, clip=1000):
    """

    At x = 

    """

    ln = tf.math.log
    pow = tf.math.pow

    def g(x):
        return ((x+1.) * ln(step_power) * pow(decay_power, -(ln(x+1.) / ln(step_power)))) / ln(decay_power/step_power)

    def center(f):
        return lambda x: f(x) - f(0.)

    def make_symmetric(f):
        return lambda x: tf.math.sign(x) * f(tf.math.abs(x))

    def do_clip(f):
        return lambda x: x if clip is None else tf.clip_by_value(f(x), -clip, clip)

    def int_float_int(f):
        return lambda x: tf.cast(f(tf.cast(x, tf.float32)), tf.int32)

    return int_float_int(do_clip(make_symmetric(center(g))))


class MeinMix(layers.Layer):

    """
    Hacky keras-specfic impl of EinMix
    """

    def __init__(self, in_shape, out_shape, regularizer, name="mix") -> None:
        super().__init__(name=name)

        self.in_shape = in_shape
        self.out_shape = out_shape

        in_shape_pattern = " ".join(in_shape.keys())
        out_shape_pattern = " ".join(out_shape.keys())

        self.out_shape_len = reduce(lambda x, y: x * y, out_shape.values())

        self.in_rearrange = f"... {in_shape_pattern} -> ... ({in_shape_pattern})"
        self.out_rearrange = f"... ({out_shape_pattern}) -> ... {out_shape_pattern}"

        self.dense = layers.Dense(
            self.out_shape_len, use_bias=False, name=name, kernel_regularizer=regularizer)

    def call(self, embd):

        embd = ein.rearrange(embd, self.in_rearrange, **self.in_shape)

        embd = self.dense(embd)

        embd = ein.rearrange(embd, self.out_rearrange, **self.out_shape)

        return embd


class RelativePositionEmbedding(layers.Layer):

    @tf_scope
    def __init__(self, max_rel_embd, D, name="rel_embd"):
        """
        If the computation only ever uses
        relative attention in one direction (eg. incremental multi-query layers with causal masking),
        then allow_negative can be set to false to halve memory use.
        """
        super().__init__(name=name)
        self.max_rel_embd = max_rel_embd
        self.embedding = tf.keras.layers.Embedding(
            self.max_rel_embd * 2 - 1, D)

        # every power of ten, double the width of the buckets.
        # starts at 1 bucket per 1 distance
        # reaches 1 bucket per 2 distance at relative distance=10
        # reaches 1 bucket per 4 distance at relative distance=100
        self.bucket_scale_fn = bucket_scale_fn(
            decay_power=2., step_power=10., clip=(self.max_rel_embd-1))

    @tf_scope
    def embeddings(self):
        indices = tf.range(self.max_rel_embd * 2 - 1)
        return self.embedding(indices)


class RelativePositionProjectionLookup(layers.Layer):

    @tf_scope
    def __init__(self, rpe: RelativePositionEmbedding, projection: MeinMix, name="rel_projection"):
        super().__init__(name=name)
        self.rpe = rpe
        self.projection = projection

    def call(self, relative_pos):
        rel_embd = self.rpe.embeddings()
        projection = self.projection(rel_embd)
        rel_proj = tf.gather(projection, relative_pos, axis=0)

        return rel_proj

def nested_tensorshape(state):
    if tf.is_tensor(state):
        return tf.TensorShape([state.shape[0], None, state.shape[2]])
    elif isinstance(state, list):
        return [nested_tensorshape(v) for v in state]
    elif isinstance(state, dict):
        return {k: nested_tensorshape(v) for k, v in state.items()}
    else:
        raise ValueError(f"Unsupported type {type(state)}")
class IRMQA(layers.Layer):
    """
    Incremental Relative Multi-Query Attention.

    Multi-query (MQ) attention (as opposed to multi-head attention)
    has a number of query projections but only single, shared key and value
    projections. This means that key/value need not be re-computed when
    performing auto-regressive inference, reducing the time complexity of
    inference significantly.

    Caching the key and value projections is called incremental inference.

    This layer also supports relative position encoding.
    """

    @tf_scope
    def __init__(self, QK, V, D, H, rpe, regularizer, name="irmqa"):
        super().__init__(name=name)

        self.QK = QK
        self.V = V
        self.D = D
        self.H = H

        self.softmax = deberta.TFDebertaXSoftmax(axis=-1)
        self.dropout = deberta.TFDebertaStableDropout(0.1)
        self.pos_dropout = deberta.TFDebertaStableDropout(0.1)

        self.wv = MeinMix(
            in_shape={"d": self.D},
            out_shape={"v": self.V},
            name="wv",
            regularizer=regularizer,
        )

        # self.wo = EinMix('b h m v -> b m d',
        #                     weight_shape='h v d', v=self.V, h=self.H, d=self.D)
        self.wo = MeinMix(
            in_shape={"h": self.H, "v": self.V},
            out_shape={"d": self.D},
            name="wo",
            regularizer=regularizer,
        )

        # self.wqc = EinMix('b m d -> b h m qk',
        #                   weight_shape='d h qk', d=self.D, h=self.H, qk=self.QK)
        self.wqc = MeinMix(
            in_shape={"d": self.D},
            out_shape={"h": self.H, "qk": self.QK},
            name="wqc",
            regularizer=regularizer,
        )
        # k & v have m dimension first so they can be put into TensorArray
        # self.wkc = EinMix('b n d -> b n qk',
        #                   weight_shape='d qk', d=self.D, qk=self.QK)
        self.wkc = MeinMix(
            in_shape={"d": self.D},
            out_shape={"qk": self.QK},
            name="wkc",
            regularizer=regularizer,
        )

        self.using_relative_position = rpe is not None
        if self.using_relative_position:

            # self.wqp = RelativePositionProjectionLookup(
            #     rpe=rpe,
            #     pattern='r d -> r h qk',
            #     weight_shape='d h qk',
            #     h=H,
            #     d=D,
            #     qk=QK,
            # )
            self.wqp = RelativePositionProjectionLookup(
                rpe=rpe,
                projection=MeinMix(
                    in_shape={"d": self.D},
                    out_shape={"h": self.H, "qk": self.QK},
                    name="wqp_proj",
                    regularizer=regularizer,
                ),
                name="wqp",
            )
            # self.wkp = RelativePositionProjectionLookup(
            #     rpe=rpe,
            #     pattern='r d -> r qk',
            #     weight_shape='d qk',
            #     d=D,
            #     qk=QK,
            # )
            self.wkp = RelativePositionProjectionLookup(
                rpe=rpe,
                projection=MeinMix(
                    in_shape={"d": self.D},
                    out_shape={"qk": self.QK},
                    name="wkp_proj",
                    regularizer=regularizer,
                ),
                name="wkp",
            )

    def create_state(self, batch_size):
        state = {
            "v_state": tf.zeros([batch_size, 0, self.V], name="v_state"),
            "kc_state": tf.zeros([batch_size, 0, self.QK], name="kc_state"),
        }
        if self.using_relative_position:
            state = state | {
                "ki_state": tf.zeros([batch_size, 0], name="ki_state"),
            }
        return state

    @tf_scope
    def call(self, kv_embd, q_embd, mask_type, state, write_state, kv_idxs=None, q_idxs=None):

        if self.using_relative_position and (kv_idxs is None or q_idxs is None):
            raise Exception(
                "Must provide kv_idxs and q_idxs when using relative position encoding")

        # Mkv is the number of new inputs this iteration, the results of which will be cached.
        #    Might be M=N, or might be M=1.
        # Mq is the number of queries this iteration.
        # N is the total number of inputs so far, including any new inputs. (so N >= Mkv)

        # kv_embd is [B, Mkv, D]
        # rel_pos is [B, Mq, N] (int32)
        # q_embd is [B, Mq, D]. If q_embd is None, it is assumed to be kv_embd, so
        #                       Mq == Mkv
        # mask is [Mq, N]
        # returns [B, Mq, D]
        # and increases N_t to N_{t+1}= N_t + Mkv

        v = self.wv(kv_embd)
        kc = self.wkc(kv_embd)

        i = tf.constant(0, tf.int32)

        if state is not None:
            kc = tf.concat([state["kc_state"], kc], axis=1)
            v = tf.concat([state["v_state"], v], axis=1)
            if write_state:
                state["kc_state"] = kc
                state["v_state"] = v

        # c2c
        qc = self.wqc(q_embd)

        # kc has batch dim second because of tensor array
        attn_logits = tf.einsum('bmhk,bnk->bmhn', qc, kc)

        # if using relative positions, use disentangled attention
        if self.using_relative_position:

            if state is not None:
                kv_idxs = tf.concat([state["ki_state"], kv_idxs], axis=1)
                if write_state:
                    state["ki_state"] = kv_idxs

            rel_pos = q_idxs[:, :, None] - kv_idxs[:, None, :]

            # p2c
            qp = self.wqp(rel_pos)
            # kc

            # h second last in this one because of embedding lookup
            # kc has batch dim second because of tensor array
            attn_logits += tf.einsum('bmnhk,bnk->bmhn', qp, kc)

            # c2p
            # qc
            kp = self.wkp(rel_pos)
            attn_logits += tf.einsum('bmhk,bmnk->bmhn', qc, kp)

            n_attn_types = 3.
        else:
            n_attn_types = 1.

        # make mask
        b = tf.shape(attn_logits)[0]
        # h
        m = tf.shape(attn_logits)[2]
        n = tf.shape(attn_logits)[3]
        if mask_type == "causal":
            mask = encoders.causal_attention_mask(b, m, n)
        elif mask_type == "none":
            mask = encoders.all_attention_mask(b, m, n)
        else:
            raise Exception("Unknown mask type: {}".format(mask_type))
        mask = mask[:, None, :, :]  # add head dim

        scale = 1. / tf.sqrt(self.D * n_attn_types)
        attn_logits *= scale
        attn_logits = self.dropout(attn_logits)        

        attn_weights = self.softmax(attn_logits, mask=mask)

        # v has batch dim second because of tensor array
        v_h = tf.einsum('bmhn,bnv->bmhv', attn_weights, v)  # value-per-head
        v_o = self.wo(v_h)  # project to output dimension
        return state, v_o


class IRMQALayer(layers.Layer):
    """
    MQA, Intermediate and Output parts.
    """

    @tf_scope
    def __init__(self, cfg, rpe, regularizer, name="irmqa"):
        super().__init__(name=name)
        self.irmqa = IRMQA(
            QK=cfg.qk_dim,
            V=cfg.v_dim,
            D=cfg.embd_dim,
            H=cfg.n_heads,
            rpe=rpe,
            regularizer=regularizer,
        )
        self.intermediate = deberta.TFDebertaIntermediate(
            intermediate_size=cfg.intermediate_size,
            initializer_range=cfg.initializer_range,
            hidden_act=cfg.hidden_act,
        )
        self.output_layer = deberta.TFDebertaOutput(
            initializer_range=cfg.initializer_range,
            hidden_size=cfg.embd_dim,
            hidden_dropout_prob=cfg.hidden_dropout_prob,
            layer_norm_eps=cfg.layer_norm_eps,
        )

    def create_state(self, batch_size):
        return self.irmqa.create_state(batch_size)

    @tf_scope
    def call(self, kv_embd, q_embd, mask_type, state, write_state, kv_idxs=None, q_idxs=None):
        
        # embd is same shape as q_embd
        state, embd = self.irmqa(
            kv_embd=kv_embd,
            q_embd=q_embd,
            kv_idxs=kv_idxs,
            q_idxs=q_idxs,
            mask_type=mask_type,
            state=state,
            write_state=write_state,
        )
        embd = self.intermediate(embd)
        embd = self.output_layer(embd, q_embd)

        return state, embd


class IRMQASelfEncoder(layers.Layer):
    """
    Multi-layer self-attention encoder for IRMQA layers.
    """

    @tf_scope
    def __init__(self, cfg, rpe, regularizer, name="hand_enc") -> None:
        super().__init__(name=name)
        self.cfg = cfg
        self.n_layers = cfg.n_layers
        self.embd_dim = cfg.embd_dim

        self.irmqa_layers = [
            IRMQALayer(cfg, rpe, regularizer, name=f"irmqa_{i}") for i in range(self.n_layers)
        ]
    
    def create_state(self, batch_size):
        return [layer.create_state(batch_size) for layer in self.irmqa_layers]

    @tf_scope
    def call(self, embd, idxs, mask_type, state, write_state):

        for i in range(self.n_layers):

            if state is None:
                s = None
            else:
                s = state[i]

            s, embd = self.irmqa_layers[i](
                kv_embd=embd,
                q_embd=embd,
                kv_idxs=idxs,
                q_idxs=idxs,
                mask_type=mask_type,
                state=s,
                write_state=write_state,
            )

            if state is not None:
                s[i] = s

        return state, embd




class MultiJointDecoder(layers.Layer):
    """
    Decodes all joints of a hand in any valid topological order.
    """

    @tf_scope
    def __init__(self, cfg, name="joint_decoder"):
        super().__init__(name=name)

        self.cfg = cfg

        self.decoder = IRMQALayer(cfg, rpe=None)

    @tf_scope
    def call(self, hand_embd, cond_joint_embd, query_joint_embd):

        batch_size = tf.shape(hand_embd)[0]
        query_seq_len = tf.shape(query_joint_embd)[1]

        def cond(i, _decoder_state, _joint_embds):
            return tf.less(i, query_seq_len)
        
        def body(i, decoder_state, joint_embds):
            decoder_state, joint_embd = self.decoder(
                kv_embd=joint_embds[:, -1:, :],
                q_embd=query_joint_embd[:, i:i+1, :],
                kv_idxs=None,
                q_idxs=None,
                mask_type="none",
                state=decoder_state,
                write_state=True,
            )
            joint_embds = tf.concat([joint_embds, joint_embd], axis=1)
            return i+1, decoder_state, joint_embds

        decoder_state = self.decoder.create_state(batch_size)
        initial_kv = tf.concat([hand_embd, cond_joint_embd], axis=1) # start with kv as hand embd

        decoder_state, joint_embds = self.decoder(
            kv_embd=initial_kv,
            q_embd=query_joint_embd[:, :1, :],
            kv_idxs=None,
            q_idxs=None,
            mask_type="none",
            state=decoder_state,
            write_state=True,
        )
        _i, _states, joint_embds = tf.while_loop(
            cond,
            body,
            [
                1,
                decoder_state,
                joint_embds
            ], 
            shape_invariants=[
                tf.TensorShape([]),
                nested_tensorshape(decoder_state),
                tf.TensorShape([None, None, self.cfg.embd_dim]),
            ],
        )

        return joint_embds


class EulerAngleDecoder(layers.Layer):
    """
    Decodes euler angles into latent vectors.
    """
    @tf_scope

    def __init__(self, cfg, name="euler_decoder"):
        super().__init__(name=name)
        self.decoder = IRMQALayer(cfg, rpe=None)

    @tf_scope
    def call(self, joint_embd, euler_query_embd):

        batch_size = tf.shape(joint_embd)[0]
        
        state = self.decoder.create_state(batch_size)

        query = euler_query_embd[:, 0:1]
        state, euler_a = self.decoder(
            kv_embd=joint_embd,
            q_embd=query,
            state=state,
            write_state=True,
            mask_type="none",
        )

        query = euler_query_embd[:, 1:2]
        state, euler_b = self.decoder(
            kv_embd=euler_a,
            q_embd=query,
            state=state,
            write_state=True,
            mask_type="none",
        )

        query = euler_query_embd[:, 2:3]
        _state, euler_c = self.decoder(
            kv_embd=euler_b,
            q_embd=query,
            state=state,
            write_state=False,
            mask_type="none",
        )

        return tf.concat([euler_a, euler_b, euler_c], 1)


class HierarchicalHandPredictor(Model):

    @tf_scope
    def __init__(self, cfg, prediction_head, name="hhp"):
        super().__init__(name=name)

        self.cfg = cfg

        if cfg.l1_reg > 0 and cfg.l2_reg > 0:
            regularizer = tf.keras.regularizers.L1L2(l1=cfg.l1_reg, l2=cfg.l2_reg)
        elif cfg.l1_reg > 0:
            regularizer = tf.keras.regularizers.L1(l=cfg.l1_reg)
        elif cfg.l2_reg > 0:
            regularizer = tf.keras.regularizers.L2(l=cfg.l2_reg)
        else:
            regularizer = None

        self.mask_type = "causal" if cfg.contiguous else "none"

        self.embd_dim = cfg.embd_dim
        self.n_dof_per_joint = cfg.n_dof_per_joint

        self.rpe = RelativePositionEmbedding(
            # maximum number of inputs.
            # defines the size of the single "rel_embd" cache
            # and n_layers "rel_proj" caches
            max_rel_embd=cfg.max_rel_embd,
            D=cfg.embd_dim,
        )

        # token to represent 'empty' conditioning info
        self.begin_token_embd = layers.Embedding(
            1, cfg.embd_dim, name=f'begin_token_embd')
        self.hand_embd = layers.Embedding(
            cfg.n_hands, cfg.embd_dim, name=f'hand_embd')
        self.joint_embd = layers.Embedding(
            cfg.n_joints_per_hand, cfg.embd_dim, name=f'joint_embd')
        self.euler_embd = layers.Embedding(
            cfg.n_dof_per_joint, cfg.embd_dim, name=f'euler_embd')

        self.hand_angles_embd = layers.Dense(
            cfg.embd_dim, name=f'frame_angle_embd', kernel_regularizer=regularizer)
        self.joint_angles_embd = layers.Dense(
            cfg.embd_dim, name=f'single_angle_embd', kernel_regularizer=regularizer)

        self.hand_encoder = IRMQASelfEncoder(cfg | cfg.hand_encoder, rpe=self.rpe, regularizer=regularizer)
        self.hand_decoder = IRMQALayer(cfg | cfg.hand_decoder, rpe=self.rpe, regularizer=regularizer, name="hand_dec")
        self.joint_decoder = MultiJointDecoder(cfg | cfg.joint_decoder, regularizer=regularizer)
        self.dof_decoder = EulerAngleDecoder(cfg | cfg.dof_decoder, regularizer=regularizer)

        self.prediction_head = prediction_head
    
    def create_state(self, batch_size):
        return {
            "hand_enc_state": self.hand_encoder.create_state(batch_size),
            "hand_dec_state": self.hand_decoder.create_state(batch_size),
        }

    @tf.function
    def call(self, inputs, state=None, write_state=False):

        cond_hand_vecs = inputs["cond_hand_vecs"]
        cond_fh_idxs = inputs["cond_fh_idxs"]
        query_fh_idxs = inputs["query_fh_idxs"]
        cond_joint_vecs = inputs["cond_joint_vecs"]
        cond_j_idxs = inputs["cond_j_idxs"]
        query_j_idxs = inputs["query_j_idxs"]

        batch_size = tf.shape(query_j_idxs)[0]
        hand_query_len = tf.shape(query_j_idxs)[1]
        joint_query_len = tf.shape(query_j_idxs)[2]

        begin_token = self.begin_token_embd(
            tf.zeros([batch_size, 1], dtype=tf.float32))
        begin_token_frame_idx = tf.tile(tf.constant(
            [[0]], dtype=tf.int32), [batch_size, 1])

        cond_frame_idxs = cond_fh_idxs[..., 0]
        cond_hand_idxs = cond_fh_idxs[..., 1]
        hand_embd = self.hand_angles_embd(
            cond_hand_vecs) + self.hand_embd(cond_hand_idxs)

        hand_embd = tf.concat([
            begin_token,
            hand_embd,
        ], axis=1)
        cond_frame_idxs = tf.concat([
            begin_token_frame_idx,
            cond_frame_idxs,
        ], axis=1)

        if state is None:
            hand_enc_state = None
            hand_dec_state = None
        else:
            hand_enc_state = state["hand_enc_state"]
            hand_dec_state = state["hand_dec_state"]

        # hand encoder
        hand_enc_state, encoded_hand_embd = self.hand_encoder(
            embd=hand_embd,
            idxs=cond_frame_idxs,
            mask_type=self.mask_type,
            state=hand_enc_state,
            write_state=write_state,
        )

        # hand decoder
        query_frame_idxs = query_fh_idxs[..., 0]
        query_hand_idxs = query_fh_idxs[..., 1]
        hand_query_embd = self.hand_embd(query_hand_idxs)
        hand_dec_state, new_hand_embd = self.hand_decoder(
            kv_embd=encoded_hand_embd,
            kv_idxs=cond_frame_idxs,
            q_embd=hand_query_embd,
            q_idxs=query_frame_idxs,
            mask_type=self.mask_type,
            state=hand_dec_state,
            write_state=write_state,
        )

        if state is not None:
            state["hand_enc_state"] = hand_enc_state
            state["hand_dec_state"] = hand_dec_state

        # joint decoder
        joint_cond_embd = self.joint_angles_embd(cond_joint_vecs) + self.joint_embd(cond_j_idxs)
        joint_cond_embd = ein.rearrange(joint_cond_embd, 'b fh j e -> (b fh) j e')
        joint_query_embd = self.joint_embd(query_j_idxs)
        joint_query_embd = ein.rearrange(joint_query_embd, 'b fh j e -> (b fh) j e')
        new_hand_embd = ein.rearrange(new_hand_embd, 'b fh e -> (b fh) 1 e')

        new_joint_embds = self.joint_decoder(
            new_hand_embd,
            joint_cond_embd,
            joint_query_embd,
        )

        # new_joint_embds = tf.ensure_shape(new_joint_embds, [batch_size * query_fh_idxs.shape[1], query_j_idxs.shape[2], None])

        # euler angle decoder
        dof_query_idxs = tf.constant([0, 1, 2], dtype=tf.int32)
        dof_query_idxs = tf.tile(dof_query_idxs[None, None, None, :], [
                                 batch_size, hand_query_len, joint_query_len, 1])
        dof_query_embd = self.euler_embd(dof_query_idxs)

        dof_query_embd = ein.rearrange(
            dof_query_embd, 'b fh j d e -> (b fh j) d e')
        new_joint_embds = ein.rearrange(
            new_joint_embds, 'bfh j e -> (bfh j) 1 e')

        # tf.ensure_shape(dof_query_embd, [batch_size * hand_query_len * joint_query_len, 3, 102])
        # tf.ensure_shape(new_joint_embds, [batch_size * hand_query_len * joint_query_len, 1, 102])
        new_euler_embds = self.dof_decoder(
            new_joint_embds,
            dof_query_embd,
        )

        # output_params = self.dof_to_output_params(new_euler_embds)

        # n_params_per_dof = output_params.shape[-1]
        # n_dof_per_joint = 3
        output_embd = tf.reshape(
            new_euler_embds,
            [
                batch_size,
                hand_query_len * joint_query_len * self.n_dof_per_joint,
                self.embd_dim
            ]
        )
        # output_embd = ein.rearrange(
        #     new_euler_embds,
        #     '(b fh j) d e -> b fh j d e',
        #     b=query_j_idxs.shape[0],
        #     fh=query_j_idxs.shape[1],
        #     j=query_j_idxs.shape[2],
        #     d=self.n_dof_per_joint,
        #     e=self.embd_dim,
        # )

        output_params = self.prediction_head(output_embd)

        return {
            "state": state,
            "output": output_params
        }

class IRMQASelfCausal(layers.Layer):

    def __init__(self, cfg, rpe, regularizer, name="irmqa_dec") -> None:
        super().__init__(name=name)

        self.irmqa_layers = [
            IRMQALayer(cfg, rpe, regularizer, name=f"irmqa_layers_{i}")
            for i in range(cfg.n_layers)
        ]
    
    def call(self, embd, idxs):
        for layer in self.irmqa_layers:
            _state, embd = layer(kv_embd=embd, q_embd=embd, mask_type="causal", state=None, write_state=None, kv_idxs=idxs, q_idxs=idxs)
        return None, embd

class IRMQACrossCausal(layers.Layer):

    def __init__(self, cfg, rpe, regularizer, name="irmqa_dec") -> None:
        super().__init__(name=name)

        self.irmqa_layers = [
            IRMQALayer(cfg, rpe, regularizer, name=f"irmqa_layers_{i}")
            for i in range(cfg.n_layers)
        ]
    
    def call(self, kv_embd, kv_idxs, q_embd, q_idxs):
        embd = q_embd
        for layer in self.irmqa_layers:
            _state, embd = layer(kv_embd=kv_embd, q_embd=embd, mask_type="causal", state=None, write_state=None, kv_idxs=kv_idxs, q_idxs=q_idxs)
        return None, embd

class DecoderOnly(tf.keras.Model):

    @tf_scope
    def __init__(self, cfg, prediction_head, name="decoder_only"):
        super().__init__(name=name)
        self.cfg = cfg

        if cfg.l1_reg > 0 and cfg.l2_reg > 0:
            regularizer = tf.keras.regularizers.L1L2(l1=cfg.l1_reg, l2=cfg.l2_reg)
        elif cfg.l1_reg > 0:
            regularizer = tf.keras.regularizers.L1(l=cfg.l1_reg)
        elif cfg.l2_reg > 0:
            regularizer = tf.keras.regularizers.L2(l=cfg.l2_reg)
        else:
            regularizer = None
        
        self.prediction_head = prediction_head
        self.angle_embd = layers.Dense(cfg.embd_dim, name="angle_embd", kernel_regularizer=regularizer)
        self.hand_embd = layers.Embedding(cfg.n_hands, cfg.embd_dim)
        self.frame_rel_embd = RelativePositionEmbedding(cfg.max_rel_embd, cfg.embd_dim)
        self.frame_abs_embd = layers.Embedding(8000, cfg.embd_dim) # TODO: set to max(dataset lengths) instead of 8000
        self.angle_unembd = layers.Dense(cfg.n_joints_per_hand * cfg.n_dof_per_joint * cfg.embd_dim, name="angle_unembd", kernel_regularizer=regularizer)
        self.decoder = IRMQASelfCausal(cfg | cfg.decoder, self.frame_rel_embd, regularizer=regularizer, name="dec")

    def call(self, inputs):

        angles = inputs["input"]
        idxs = inputs["input_idxs"]

        # produce a bunch of rotations to make the model's job easier
        n = self.cfg.n_rotations
        scale = (tau / 4.) * (1. / n) # only need to produce rotations up to tau/4, because the model can easily invert angles
        offset = tf.range(n, dtype=tf.float32) * tf.constant([scale])
        angles = angles[:, :, :, :, None] + offset[None, None, None, None, :]
        angles = tf.stack([tf.sin(angles), tf.cos(angles)], axis=-1)

        angles = ein.rearrange(angles, 'b fh j d rot sincos -> b fh (j d rot sincos)')

        frame_idxs = idxs[:, :, 0]
        hand_idxs = idxs[:, :, 1]

        embd = self.angle_embd(angles) + self.hand_embd(hand_idxs) #+ self.frame_abs_embd(frame_idxs)

        _, embd = self.decoder(embd, frame_idxs)

        embd = embd + self.frame_abs_embd(frame_idxs)

        latents = self.angle_unembd(embd)

        latents = ein.rearrange(latents, 'b fh (j d e) -> b (fh j d) e', j=self.cfg.n_joints_per_hand, d=self.cfg.n_dof_per_joint, e=self.cfg.embd_dim)

        return {
            "output": self.prediction_head(latents),
        }


class EncoderDecoder(tf.keras.Model):

    @tf_scope
    def __init__(self, cfg, prediction_head, name="enc_dec"):
        super().__init__(name=name)
        self.cfg = cfg

        if cfg.l1_reg > 0 and cfg.l2_reg > 0:
            regularizer = tf.keras.regularizers.L1L2(l1=cfg.l1_reg, l2=cfg.l2_reg)
        elif cfg.l1_reg > 0:
            regularizer = tf.keras.regularizers.L1(l=cfg.l1_reg)
        elif cfg.l2_reg > 0:
            regularizer = tf.keras.regularizers.L2(l=cfg.l2_reg)
        else:
            regularizer = None
        
        self.prediction_head = prediction_head
        self.angle_embd = layers.Dense(cfg.embd_dim, name="angle_embd", kernel_regularizer=regularizer)
        self.hand_embd = layers.Embedding(cfg.n_hands, cfg.embd_dim)
        self.frame_rel_embd = RelativePositionEmbedding(cfg.max_rel_embd, cfg.embd_dim)
        self.frame_abs_embd = layers.Embedding(8000, cfg.embd_dim) # TODO: set to max(dataset lengths) instead of 8000
        self.angle_unembd = layers.Dense(cfg.embd_dim * cfg.n_joints_per_hand * cfg.n_dof_per_joint, name="angle_unembd", kernel_regularizer=regularizer)
        self.encoder = IRMQASelfCausal(cfg | cfg.encoder, self.frame_rel_embd, regularizer=regularizer, name="enc")
        self.hand_embd_2 = layers.Embedding(cfg.n_hands, cfg.embd_dim)
        self.frame_rel_embd_2 = RelativePositionEmbedding(cfg.max_rel_embd, cfg.embd_dim)
        self.frame_abs_embd_2 = layers.Embedding(8000, cfg.embd_dim) # TODO: set to max(dataset lengths) instead of 8000
        self.n_ahead_embeddings = layers.Embedding(20, cfg.embd_dim)
        self.decoder = IRMQACrossCausal(cfg | cfg.decoder, self.frame_rel_embd_2, regularizer=regularizer, name="dec")

    def call(self, inputs):

        angles = inputs["input"]
        inp_idxs = inputs["input_idxs"]
        tar_idxs = inputs["target_idxs"]
        n_ahead = inputs["n_ahead"]

        # produce a bunch of rotations to make the model's job easier
        n = self.cfg.n_rotations
        scale = (tau / 4.) * (1. / n) # only need to produce rotations up to tau/4, because the model can easily invert angles
        offset = tf.range(n, dtype=tf.float32) * tf.constant([scale])
        angles = angles[:, :, :, :, None] + offset[None, None, None, None, :]
        angles = tf.stack([tf.sin(angles), tf.cos(angles)], axis=-1)

        angles = ein.rearrange(angles, 'b fh j d rot sincos -> b fh (j d rot sincos)')

        inp_frame_idxs = inp_idxs[:, :, 0]
        inp_hand_idxs = inp_idxs[:, :, 1]
        tar_frame_idxs = tar_idxs[:, :, 0]
        tar_hand_idxs = tar_idxs[:, :, 1]

        inp_embd = self.angle_embd(angles)# + self.hand_embd(inp_hand_idxs) + self.frame_abs_embd(inp_frame_idxs)
        n_ahead = tf.ones_like(inp_frame_idxs[0], dtype=tf.int32)
        tar_embd = self.n_ahead_embeddings(n_ahead, ) #+ self.hand_embd_2(tar_hand_idxs) + self.frame_abs_embd_2(tar_frame_idxs)

        _, embd = self.encoder(embd=inp_embd, idxs=inp_frame_idxs)
        _, embd = self.decoder(kv_embd=embd, kv_idxs=inp_frame_idxs, q_embd=tar_embd, q_idxs=tar_frame_idxs)

        latents = self.angle_unembd(embd)

        latents = ein.rearrange(latents, 'b fh (j d e) -> b (fh j d) e', j=self.cfg.n_joints_per_hand, d=self.cfg.n_dof_per_joint)

        return {
            "output": self.prediction_head(latents),
        }

def random_subset(options, n, seed=None):
    options = tf.random.shuffle(options, seed=seed)
    return options[:n]


@tf_scope
def weighted_random_n(n_max, D=2., B=10., seed=None):
    """
    Picks random indices from [0, n_max) with skewed distribution
    towards the lower numbers.
    p(index i) = 1/D * p(index Di) = 1/D**2 * p(index D**2i)
    The probability of each index decreases by D every power of b
    """

    B, D, n_max = tf.cast(B, tf.float32), tf.cast(
        D, tf.float32), tf.cast(n_max, tf.float32)

    def logb1p(b, x): return tf.math.log1p(x)/tf.math.log(b)
    # probabilities
    # probabilities = l1_norm(powf(D, -logb1p(B, tf.range(n_max))))
    # log probabilities base e
    log_probabilities = -logb1p(B, tf.range(n_max))*tf.math.log(D)

    return tf.random.categorical(log_probabilities[None, :], 1, seed=seed)[0, 0]

@tf_scope
def random_joint_order(n_joints_per_hand, seed=None):
    """
    Produce a random selection of N joint indices, such that they
    still obey the hierarchy of the hand skeleton. i.e. the
    wrist pos must be produced first, then the joints of each
    finger must be produced in order (although they might be
    interspersed.)

    Uses tensorflow only, in order to run in data pipeline.
    """

    joint_idxs = tf.ragged.constant([
        [0],  # wrist
        [1, 2, 3, 4],
        [5, 6, 7],
        [8, 9, 10],
        [11, 12, 13],
        [14, 15, 16],
    ])
    finger_order = tf.TensorArray(tf.int32, size=n_joints_per_hand)
    finger_order = finger_order.write(0, 0)  # wrist

    i_finger = 1  # skip wrist
    i_joint = 1
    while i_finger < 6 and i_joint < n_joints_per_hand:
        n_joints_this_finger = len(joint_idxs[i_finger])
        i = 0
        while i < n_joints_this_finger and i_joint < n_joints_per_hand:
            finger_order.write(i_joint, i_finger)
            i_joint += 1
            i += 1

    finger_order = finger_order.stack()
    finger_order = tf.random.shuffle(finger_order, seed=seed)
    n_fingers = i_finger

    per_finger_i = tf.TensorArray(tf.int32, n_fingers, element_shape=[])
    joint_order = tf.TensorArray(tf.int32, n_joints_per_hand, element_shape=[])
    i = 0
    for finger in finger_order:
        i_knuckle = per_finger_i.read(finger)
        joint_order = joint_order.write(i, joint_idxs[finger, i_knuckle])
        i += 1
        per_finger_i = per_finger_i.write(finger, i_knuckle + 1)

    joint_order = joint_order.stack()
    joint_order = tf.ensure_shape(joint_order, [n_joints_per_hand])

    return joint_order



@tf_scope
def get_chunk(cfg, chunk_size, angles, chunk_mode, seed=None):
    angles = angles

    angles = ein.rearrange(
        angles, 'f h (j d) -> f h j d', j=cfg.n_joints_per_hand, d=cfg.n_dof_per_joint)
    n_frames = tf.shape(angles)[0]
    fh_idxs = utils.multidim_indices([n_frames, cfg.n_hands], flatten=True)
    if chunk_mode == "overlapping":
        i = tf.random.uniform(
            [], minval=0, maxval=n_frames-chunk_size, seed=seed, dtype=tf.int32)
        idxs = tf.concat([fh_idxs[i:i+chunk_size//2],
                            fh_idxs[i+1:i+chunk_size//2+1]], axis=0)
    elif chunk_mode == "simple":
        i = tf.random.uniform(
            [], minval=0, maxval=n_frames-chunk_size, seed=seed, dtype=tf.int32)
        idxs = fh_idxs[i:i+chunk_size]
    else:
        idxs = random_subset(fh_idxs, chunk_size, seed=seed)
    hands = tf.gather_nd(angles, idxs)

    return hands, idxs

@tf_scope
def flat_vector_batched_random_chunk(cfg, x, seed=None):

    if cfg.random_ahead:
        n_ahead = weighted_random_n(20, 2., 3., seed=seed)
    else:
        n_ahead = tf.constant(1, tf.int32)

    chunk_size = cfg.n_hand_vecs + n_ahead

    vecs, idxs = tf.map_fn(
        lambda a: get_chunk(cfg, chunk_size, a, chunk_mode="simple", seed=seed),
        x["angles"],
        fn_output_signature=(
            tf.TensorSpec(shape=[None, cfg.n_joints_per_hand,
                          cfg.n_dof_per_joint], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
        ),
        name="hand_map",
        parallel_iterations=10,
    )

    n_frames = tf.shape(vecs)[1]
    input = vecs[:, :n_frames-n_ahead, :, :]
    input_idxs = idxs[:, :n_frames-n_ahead, :]
    target = vecs[:, n_ahead:, :, :]
    target_idxs = idxs[:, n_ahead:, :]

    input = ein.rearrange(input, 'b fh j d -> b fh j d')
    target = ein.rearrange(target, 'b fh j d -> b (fh j d)')

    return (
        x | {
            "input": input,
            "input_idxs": input_idxs,
            "target_idxs": target_idxs,
            "n_ahead": n_ahead,
        },
        {
            "target_output": target,
        },
    )

@tf.function
@tf_scope
def hierarchical_batched_random_chunk(cfg, x, seed=None):

    if cfg.contiguous:
        chunk_mode = "overlapping"
    else:
        chunk_mode = "random"

    hands, fh_idxs = tf.map_fn(
        lambda a: get_chunk(cfg, cfg.n_hand_vecs, a, chunk_mode, seed),
        x["angles"],
        fn_output_signature=(
            tf.TensorSpec(shape=[None, cfg.n_joints_per_hand,
                          cfg.n_dof_per_joint], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
        ),
        name="hand_map",
        parallel_iterations=10,
    )
    if cfg.contiguous:
        assert cfg.n_hand_vecs % 2 == 0, "n_hand_vecs must be even if cfg.contiguous"
        n_cond_hand_vecs = cfg.n_hand_vecs//2
    else:
        # 100 times more likely to have 0 than 50
        n_cond_hand_vecs = weighted_random_n(
            cfg.n_hand_vecs, D=100., B=cfg.n_hand_vecs, seed=seed)
    n_query_hands = cfg.n_hand_vecs - n_cond_hand_vecs
    cond_fh_idxs = fh_idxs[:, :n_cond_hand_vecs]
    cond_hand_vecs = ein.rearrange(
        hands[:, :n_cond_hand_vecs], 'b fh j d -> b fh (j d)')

    query_fh_idxs = fh_idxs[:, n_cond_hand_vecs:]
    target_hands = hands[:, n_cond_hand_vecs:]

    j_idxs = tf.map_fn(
        lambda _i: random_joint_order(cfg.n_joints_per_hand, seed=seed),
        tf.range(cfg.batch_size * n_query_hands),
        name="joint_map",
        parallel_iterations=10,
    )
    tf.ensure_shape(
        j_idxs, [cfg.batch_size * n_query_hands, cfg.n_joints_per_hand])
    j_idxs = ein.rearrange(j_idxs, '(b qfh) j -> b qfh j', b=cfg.batch_size)

    # always have at least 1 query joint
    max_n_cond_joints = cfg.n_joints_per_hand - 1
    if max_n_cond_joints == 0:
        n_cond_joint_vecs = 0
    else:
        n_cond_joint_vecs = weighted_random_n(
            max_n_cond_joints, D=10., B=cfg.n_joints_per_hand, seed=seed)
    cond_j_idxs = j_idxs[:, :, :n_cond_joint_vecs]
    query_j_idxs = j_idxs[:, :, n_cond_joint_vecs:]

    tf.ensure_shape(cond_j_idxs, [cfg.batch_size, n_query_hands, n_cond_joint_vecs])
    tf.ensure_shape(query_j_idxs, [cfg.batch_size, n_query_hands, cfg.n_joints_per_hand - n_cond_joint_vecs])

    cond_joint_vecs = tf.gather(target_hands, cond_j_idxs, batch_dims=2)
    target_joint_vecs = tf.gather(target_hands, query_j_idxs, batch_dims=2)
    target_joint_vecs = ein.rearrange(target_joint_vecs, 'b fh j d -> b (fh j d)')
    return (
        x | {
            "cond_hand_vecs": cond_hand_vecs,
            "cond_fh_idxs": cond_fh_idxs,
            "query_fh_idxs": query_fh_idxs,
            "cond_joint_vecs": cond_joint_vecs,
            "cond_j_idxs": cond_j_idxs,
            "query_j_idxs": query_j_idxs,
        },
        {
            "target_output": target_joint_vecs,
        },
    )


def hierarchical_dataset(cfg, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset):

    train_dataset = train_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=cfg.batch_size))
    train_dataset = train_dataset.map(
        lambda x: hierarchical_batched_random_chunk(cfg, x))

    test_dataset = test_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=cfg.test_batch_size))
    test_dataset = test_dataset.map(
        lambda x: hierarchical_batched_random_chunk(cfg, x, seed=1234))

    val_dataset = val_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=cfg.batch_size))
    val_dataset = val_dataset.map(
        lambda x: hierarchical_batched_random_chunk(cfg, x, seed=1234))

    return train_dataset, test_dataset, val_dataset

def flat_dataset(cfg, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset):

    train_dataset = train_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=cfg.batch_size))
    train_dataset = train_dataset.map(
        lambda x: flat_vector_batched_random_chunk(cfg, x))

    test_dataset = test_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=cfg.test_batch_size))
    test_dataset = test_dataset.map(
        lambda x: flat_vector_batched_random_chunk(cfg, x, seed=1234))

    val_dataset = val_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=cfg.batch_size))
    val_dataset = val_dataset.map(
        lambda x: flat_vector_batched_random_chunk(cfg, x, seed=1234))

    return train_dataset, test_dataset, val_dataset

class MyMetric(abc.ABC, layers.Layer):

    def __init__(self, name: str):
        super().__init__(name=name)

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def result(self):
        pass

    @abc.abstractmethod
    def update_state(self, i, inputs):
        pass

class RunningMean(MyMetric):
    
    @tf_scope
    def __init__(self, fn, element_shape=[], dtype=tf.float32, name="running_mean"):
        super().__init__(name=name)
        self.total = tf.Variable(initial_value=tf.zeros(element_shape, dtype=dtype), name="total", trainable=False)
        self.count = tf.Variable(0., dtype=tf.float32, name="count", trainable=False)
        self.fn = fn
    
    def reset(self):
        self.total.assign(tf.zeros_like(self.total))
        self.count.assign(0.)

    @tf_scope
    def result(self):
        return self.total / self.count

    @tf_scope
    def update_state(self, inputs):
        val = self.fn(inputs)
        self.total.assign_add(val),
        self.count.assign_add(1.)

class Rolling(MyMetric):

    @tf_scope
    def __init__(self, length, fn, element_shape=[], dtype=tf.float32, reduction_fn=tf.reduce_mean, name="rolling"):
        super().__init__(name=name)
        self.length = length
        self.reduction_fn = reduction_fn
        self.fn = fn
        self.buffer = tf.Variable(
            initial_value=tf.zeros(shape=[length] + element_shape, dtype=dtype),
            name="history",
            trainable=False,
            aggregation=tf.VariableAggregation.SUM,
            synchronization=tf.VariableSynchronization.ON_READ,
        )
        self.index = tf.Variable(
            initial_value=tf.constant(0, dtype=tf.int64),
            name="index",
            trainable=False,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            synchronization=tf.VariableSynchronization.ON_READ,
        )
    
    def reset(self):
        self.index.assign(0)
        self.buffer.assign(tf.zeros_like(self.buffer))

    @tf_scope
    def update_state(self, inputs):
        self.index.assign_add(1)
        i = self.index % self.length
        val = self.fn(inputs)
        self.buffer[i].assign(val)
    
    @tf_scope
    def result(self):
        i = tf.math.minimum(self.index, self.length)
        return self.reduction_fn(self.buffer[:i])

def train(cfg, run_name):

    cfg = cfg | cfg.dream

    if cfg.ds == "synthetic":
        cfg = cfg | cfg.ds_synthetic
        data_fn = data_tf.synthetic_data
    elif cfg.ds == "real":
        cfg = cfg | cfg.ds_real
        data_fn = data_tf.bvh_data
    else:
        raise ValueError("Unknown dataset '{}'".format(cfg.ds))

    
    if cfg.loss == "angular_mse":
        loss_fn, stat_fns, prediction_head = prediction_heads.angular(cfg)
    elif cfg.loss == "vmf_crossentropy":
        loss_fn, stat_fns, prediction_head = prediction_heads.von_mises_fisher(cfg, name="vmf")
    elif cfg.loss == "vm_atan_crossentropy":
        loss_fn, stat_fns, prediction_head = prediction_heads.von_mises_atan(cfg, name="vm_atan")
    else:
        raise ValueError("Unknown loss type '{}'".format(cfg.loss))

    print("Creating model ... ", end="", flush=True)
    if cfg.task == "flat":
        cfg = cfg | cfg.task_flat
        d_train, d_test, d_val = data_tf.tf_dataset(cfg, flat_dataset, data_fn=data_fn)
        model = DecoderOnly(cfg | cfg.model, prediction_head=prediction_head)
    elif cfg.task == "flat_query":
        cfg = cfg | cfg.task_flat_query
        d_train, d_test, d_val = data_tf.tf_dataset(cfg, flat_dataset, data_fn=data_fn)
        model = EncoderDecoder(cfg | cfg.model, prediction_head=prediction_head)
    elif cfg.task == "hierarchical":
        cfg = cfg | cfg.task_hierarchical
        d_train, d_test, d_val = data_tf.tf_dataset(cfg, hierarchical_dataset, data_fn=data_fn)
        model = HierarchicalHandPredictor(cfg | cfg.model, prediction_head=prediction_head)
    else:
        raise ValueError("Unknown task: {}".format(cfg.task))
    print("Done.")

    print("Initializing datasets ... ", end="", flush=True)
    _, _ = next(iter(d_train))
    _, _ = next(iter(d_test))
    _, _ = next(iter(d_val))
    print("Done.")
    print()

    def loss_fn_wrapper(inp):
        return tf.reduce_mean(loss_fn(inp["targets"]["target_output"], inp["outputs"]["output"]))

    optimizer = keras.optimizers.Adam(cfg.adam.lr)

    with enlighten.get_manager() as manager:

        predict_fn, predict_and_plot_fn = predict.create_predict_fn_v2(cfg, run_name, model, stat_fns)

        test_inp_data, _test_tar_data = next(iter(d_test))

        eval_fn = make_exec_loop(
            "Validation",
            model,
            d_val,
            loss_fn,
            optimizer,
            training=False,
            metrics=[
                {
                    "name": "Val Loss",
                    "metric": RunningMean(loss_fn_wrapper, name="mean_val_loss"),
                }
            ]
        )

        def checkpoint_fn(i):
            if i == 0:
                model.save(f"_models/{run_name}/model")
            else:
                model.save_weights(f"_models/{run_name}/weights_{i}")
        
        train_fn = make_exec_loop(
            "Training",
            model,
            d_train,
            loss_fn,
            optimizer,
            training=True,
            metrics=[
                {
                    "name": "Loss (10)",
                    "metric": Rolling(10, loss_fn_wrapper, name="train_loss_10"),
                },
                {
                    "name": "Loss (100)",
                    "metric": Rolling(100, loss_fn_wrapper, name="train_loss_100"),
                },
                {
                    "name": "Loss (1000)",
                    "metric": Rolling(1000, loss_fn_wrapper, name="train_loss_1000"),
                },
            ],
            callbacks=[
                {
                    "name": "Epoch",
                    "every": 1000,
                    
                    "fns": [
                        {
                            "name": "Evaluate",
                            "fn": lambda i: eval_fn(manager)[1][0],
                        },
                        {
                            "name": "Predict",
                            "fn": lambda i: predict_and_plot_fn(test_inp_data, n_frames=100, seed_len=5, id=i.numpy()),
                        },
                        {
                            "name": "Checkpoint",
                            "fn": checkpoint_fn,
                        },
                    ],
                },
            ],
        )


        print()
        print(f"Training model '{run_name}' ... ")
        print()
        stopped_early, metric_results = train_fn(manager)

        print()
        print()
        if stopped_early:
            print(f"Run '{run_name}' stopped early.")
        else:
            print(f"Run '{run_name}' completed successfully!")
        print()


def make_exec_loop(name, model, dataset, loss_fn, optimizer, training, metrics=[], callbacks=[], n_fuse_steps=10):

    use_metrics = len(metrics) > 0
    use_callbacks = len(callbacks) > 0

    def train_step_fn(batch):
        i, (inputs, targets) = batch
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = loss_fn(targets["target_output"], outputs["output"])
            loss = tf.reduce_mean(loss)
            loss += tf.reduce_mean(model.losses)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if use_metrics:
            for metric in metrics:
                metric["metric"].update_state({
                    "inputs": inputs,
                    "targets": targets,
                    "outputs": outputs,
                    "loss": loss,
                    "grads": grads,
                })
    
    def eval_step_fn(batch):
        i, (inputs, targets) = batch
        outputs = model(inputs, training=False)
        loss = loss_fn(targets["target_output"], outputs["output"])
        if use_metrics:
            for metric in metrics:
                metric["metric"].update_state({
                    "inputs": inputs,
                    "targets": targets,
                    "outputs": outputs,
                    "loss": loss,
                })

    # add indexes to the dataset
    dataset = dataset.enumerate()

    if training:
        step_fn = train_step_fn
    else:
        step_fn = eval_step_fn
    
    print()
    print(f"Compiling {name.lower()} step function ... ", end="", flush=True)
    compiled_step_fn = tf.function(
        step_fn,
        input_signature=[
            dataset.element_spec,
        ],
        reduce_retracing=True,
        # jit_compile=True,
    )
    step_fn = compiled_step_fn

    _ = step_fn(next(iter(dataset)))
    print("Done.")
    print()

    def exec_loop(manager):

        if use_metrics:
            for metric in metrics:
                metric["metric"].reset()

            metric_column_length = max(len(metric["name"]) for metric in metrics)
            metric_header_text = " | ".join(
                "{name:^{length}}".format(name=metric["name"], length=metric_column_length)
                for metric in metrics
            )
            metric_status_text = lambda: "| " + " | ".join(
                "{value: > {length}.{sf}e}".format(value=metric["metric"].result(), length=metric_column_length, sf=metric.get("sf", 4))
                for metric in metrics
            ) + " |"
            metric_header_bar = manager.status_bar(metric_header_text, leave=False, min_delta=0.01)
            metric_status_bar = manager.status_bar(metric_status_text(), leave=False, min_delta=0.01)

        n_steps = dataset.cardinality().numpy()
        if n_steps == tf.data.INFINITE_CARDINALITY or n_steps == tf.data.UNKNOWN_CARDINALITY:
            raise ValueError("Dataset has unknown cardinality.")
        
        steps_counter = manager.counter(
            total=n_steps,
            desc=f"{name} progress",
            unit="steps",
            color='skyblue',
            leave=False,
            min_delta=0.01,
        )

        if use_callbacks:
            longest_name = max(len(callback["name"]) for callback in callbacks)
            for callback in callbacks:
                callback["counter"] = manager.counter(
                    total=callback["every"],
                    desc=f"{callback['name']:<{longest_name}}",
                    unit=callback.get("unit", "steps"),
                    color=callback.get("color", "sandybrown"),
                    leave=False,
                    min_delta=0.01,
                )
        try:
            for i, batch in dataset:
                if use_callbacks:
                    for callback in callbacks:
                        if i % callback["every"] == 0:
                            for f in callback["fns"]:
                                f["fn"](i)
                            callback["counter"].count = 0
                            callback["counter"].refresh(elapsed=0.)
                step_fn((i, batch))
                if use_metrics:
                    metric_header_bar.update(metric_header_text)
                    metric_status_bar.update(metric_status_text())
                steps_counter.update()
                if use_callbacks:
                    for callback in callbacks:
                        callback["counter"].update()
            stopped_early = False
        except KeyboardInterrupt:
            stopped_early = True
        
        if use_callbacks:
            for callback in callbacks:
                callback["counter"].close()
        
        if use_metrics:
            metric_status_bar.close()
            metric_header_bar.close()
        
        steps_counter.close()
        
        return stopped_early, [metric["metric"].result() for metric in metrics]
    
    return exec_loop
            
