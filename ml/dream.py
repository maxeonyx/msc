from functools import reduce
from http.client import INSUFFICIENT_STORAGE
from mimetypes import init
from ntpath import join
from pprint import pprint
from box import Box
import einops as ein
from einops.layers.keras import EinMix
import tensorflow as tf
from ml import data_tf, decoders, encoders, utils, deberta
from tensorflow import keras
from tensorflow.keras import Input, Model, layers
from tensorflow_probability import distributions as tfd
from ml.utils import tf_scope


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


class MeinMix(tf.Module):

    """
    Hacky keras-specfic impl of EinMix
    """

    def __init__(self, in_shape, out_shape, name="mix") -> None:
        super().__init__(name=name)

        self.in_shape = in_shape
        self.out_shape = out_shape

        in_shape_pattern = " ".join(in_shape.keys())
        out_shape_pattern = " ".join(out_shape.keys())

        self.out_shape_len = reduce(lambda x, y: x * y, out_shape.values())

        self.in_rearrange = f"... {in_shape_pattern} -> ... ({in_shape_pattern})"
        self.out_rearrange = f"... ({out_shape_pattern}) -> ... {out_shape_pattern}"

        self.dense = layers.Dense(
            self.out_shape_len, use_bias=False, name=name)

    def __call__(self, embd):

        embd = ein.rearrange(embd, self.in_rearrange, **self.in_shape)

        embd = self.dense(embd)

        embd = ein.rearrange(embd, self.out_rearrange, **self.out_shape)

        return embd


class RelativePositionEmbedding(tf.Module):

    @tf_scope
    def __init__(self, max_rel_embd, D,  allow_negative=False, name="rel_embd"):
        """
        If the computation only ever uses
        relative attention in one direction (eg. incremental multi-query layers with causal masking),
        then allow_negative can be set to false to halve memory use.
        """
        super().__init__(name=name)
        self.max_rel_embd = max_rel_embd
        if allow_negative:
            self.embedding = tf.keras.layers.Embedding(
                self.max_rel_embd * 2 - 1, D)
        else:
            self.embedding = tf.keras.layers.Embedding(self.max_rel_embd, D)
        self.allow_negative = allow_negative

        # every power of ten, double the width of the buckets.
        # starts at 1 bucket per 1 distance
        # reaches 1 bucket per 2 distance at relative distance=10
        # reaches 1 bucket per 4 distance at relative distance=100
        self.bucket_scale_fn = bucket_scale_fn(
            decay_power=2., step_power=10., clip=(self.max_rel_embd-1))

    def __call__(self, relative_pos):
        if self.allow_negative:
            indices = self.bucket_scale_fn(
                relative_pos) + self.max_rel_embd - 1
        else:
            indices = self.bucket_scale_fn(relative_pos)
        return self.embedding(indices)

    @tf_scope
    def get_indices(self, relative_pos):
        """
        Log-bucketing of relative position embeddings.
        Buckets are small near 0, and grow larger with distance:
            @ x = 0, there is 1 bucket (index) per 1 distance
            @ x = 10, there is 1 bucket (index) per 2 distance
            @ x = 100, there is 1 bucket (index) per 4 distance
            etc.
        """
        return indices

    @tf_scope
    def get_all_embeddings(self):
        if self.allow_negative:
            indices = tf.range(self.max_rel_embd * 2 - 1)
        else:
            indices = tf.range(self.max_rel_embd)
        return 


class RelativePositionProjectionLookup(tf.Module):

    @tf_scope
    def __init__(self, rpe: RelativePositionEmbedding, projection: MeinMix, name="rel_projection"):
        super().__init__(name=name)
        self.rpe = rpe
        self.projection = projection
        self.cache = None

    def reset_cache(self):

    def __call__(self, relative_pos):
        if self.cache is None:
            self.reset_cache()

        indices = self.rpe.get_indices(relative_pos)
        rel_embd = self.rpe.get_all_embeddings()
        projection = self.projection(rel_embd)
        rel_proj = tf.gather(self.cache, indices, axis=0)

        return rel_proj


class IRMQA(tf.Module):
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
    def __init__(self, QK, V, D, H, rpe, name="irmqa"):
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
            name="wv"
        )

        # self.wo = EinMix('b h m v -> b m d',
        #                     weight_shape='h v d', v=self.V, h=self.H, d=self.D)
        self.wo = MeinMix(
            in_shape={"h": self.H, "v": self.V},
            out_shape={"d": self.D},
            name="wo"
        )

        # self.wqc = EinMix('b m d -> b h m qk',
        #                   weight_shape='d h qk', d=self.D, h=self.H, qk=self.QK)
        self.wqc = MeinMix(
            in_shape={"d": self.D},
            out_shape={"h": self.H, "qk": self.QK},
            name="wqc"
        )
        # k & v have m dimension first so they can be put into TensorArray
        # self.wkc = EinMix('b n d -> b n qk',
        #                   weight_shape='d qk', d=self.D, qk=self.QK)
        self.wkc = MeinMix(
            in_shape={"d": self.D},
            out_shape={"qk": self.QK},
            name="wkc"
        )

        self.v_cache = None

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
                    name="wqp_proj"
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
                    name="wkp_proj"
                ),
                name="wkp",
            )

    def reset_cache(self, batch_size):
        if self.using_relative_position:
            self.wqp.reset_cache()
            self.wkp.reset_cache()

    @tf_scope
    def __call__(self, kv_embd, q_embd, mask_type, state, write_state, kv_idxs=None, q_idxs=None):

        if state and not hasattr(self, 'v_cache'):
            raise Exception(
                "Must call build before calling calling with state=True")

        if self.using_relative_position and (kv_idxs is None or q_idxs is None):
            raise Exception(
                "Must provide kv_idxs and q_idxs when using relative position encoding")

        if state and self.v_cache is None:
            self.v_cache = tf.TensorArray(
                dtype=tf.float32,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
                element_shape=[None, None, self.V],
            )
            self.kc_cache = tf.TensorArray(
                dtype=tf.float32,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
                element_shape=[None, None, self.QK],
            )
            if self.using_relative_position:
                self.ki_cache = tf.TensorArray(
                    dtype=tf.int32,
                    size=0,
                    dynamic_size=True,
                    clear_after_read=False,
                    element_shape=[None, None],
                )

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

        if state:
            # v = ein.rearrange(v, 'b n v -> n b v')
            v = tf.transpose(v, perm=[1, 0, 2])
            # kc = ein.rearrange(kc, 'b n qk -> n b qk')
            kc = tf.transpose(kc, perm=[1, 0, 2])
            self.v_cache = self.v_cache.write(i, v)
            self.kc_cache = self.kc_cache.write(i, kc)
            v = self.v_cache.concat()
            kc = self.kc_cache.concat()
            # v = ein.rearrange(v, 'n b v -> b n v')
            v = tf.transpose(v, perm=[1, 0, 2])
            # kc = ein.rearrange(kc, 'n b qk -> b n qk')
            kc = tf.transpose(kc, perm=[1, 0, 2])

        # c2c
        qc = self.wqc(q_embd)

        # kc has batch dim second because of tensor array
        attn_logits = tf.einsum('bmhk,bnk->bmhn', qc, kc)

        # if using relative positions, use disentangled attention
        if self.using_relative_position:

            if state:
                # kv_idxs = ein.rearrange(kv_idxs, 'b n -> n b')
                kv_idxs = tf.transpose(kv_idxs, perm=[1, 0])
                self.ki_cache = self.ki_cache.write(i, kv_idxs)
                kv_idxs = self.ki_cache.concat()
                # kv_idxs = ein.rearrange(kv_idxs, 'n b -> b n')
                kv_idxs = tf.transpose(kv_idxs, perm=[1, 0])

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
        return v_o


class IRMQALayer(tf.Module):
    """
    MQA, Intermediate and Output parts.
    """

    @tf_scope
    def __init__(self, cfg, rpe, name="irmqa"):
        super().__init__(name=name)
        self.irmqa = IRMQA(
            QK=cfg.qk_dim,
            V=cfg.v_dim,
            D=cfg.embd_dim,
            H=cfg.n_heads,
            rpe=rpe,
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
        self.using_relative_position = rpe is not None

    def reset_cache(self, batch_size):
        self.irmqa.reset_cache(batch_size)

    @tf_scope
    def __call__(self, kv_embd, q_embd, mask_type, state, write_state, kv_idxs=None, q_idxs=None):

        if self.using_relative_position and (kv_idxs is None or q_idxs is None):
            raise Exception(
                "If using relative attention, must provide both kv_idxs and q_idxs.")

        embd = self.irmqa(
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

        return embd


class IRMQAEncoder(tf.Module):
    """
    Multi-layer self-attention encoder for IRMQA layers.
    """

    @tf_scope
    def __init__(self, cfg, rpe, name="hand_enc") -> None:
        super().__init__(name=name)
        self.cfg = cfg
        self.n_layers = cfg.n_layers
        self.embd_dim = cfg.embd_dim

        self.irmqa_layers = [
            IRMQALayer(cfg, rpe, name=f"irmqa_{i}") for i in range(self.n_layers)
        ]
    
    def get_config(self):
        return self.cfg

    def reset_cache(self, batch_size):
        for layer in self.irmqa_layers:
            layer.reset_cache(batch_size)

    @tf_scope
    def __call__(self, embd, idxs, mask_type, state, write_state):

        for layer in self.irmqa_layers:
            embd = layer(
                kv_embd=embd,
                q_embd=embd,
                kv_idxs=idxs,
                q_idxs=idxs,
                mask_type=mask_type,
                state=state,
                write_state=write_state,
            )

        return embd


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
    i_joint = 0
    while i_finger < 6 and i_joint < n_joints_per_hand:
        n_joints_this_finger = len(joint_idxs[i_finger])
        i = 0
        while i < n_joints_this_finger and i_joint < n_joints_per_hand:
            finger_order.write(i_joint, i_finger)
            i_joint += 1
            i += 1

    finger_order = finger_order.stack()
    finger_order = tf.random.shuffle(finger_order, seed=seed)
    n_fingers = tf.shape(finger_order)[0]

    per_finger_i = tf.TensorArray(tf.int32, n_fingers, element_shape=[])
    joint_order = tf.TensorArray(tf.int32, n_joints_per_hand)
    i = 0
    for finger in finger_order:
        i_knuckle = per_finger_i.read(finger)
        joint_order = joint_order.write(i, joint_idxs[finger, i_knuckle])
        i += 1
        per_finger_i = per_finger_i.write(finger, i_knuckle + 1)

    joint_order = joint_order.stack()
    joint_order = tf.ensure_shape(joint_order, [n_joints_per_hand])

    return joint_order


class MultiJointDecoder(tf.Module):
    """
    Decodes all joints of a hand in any valid topological order.
    """

    @tf_scope
    def __init__(self, cfg, name="joint_decoder"):
        super().__init__(name=name)

        self.cfg = cfg

        self.decoder = IRMQALayer(cfg, rpe=None)

    @tf_scope
    def __call__(self, hand_embd, cond_joint_embd, query_joint_embd):

        batch_size = tf.shape(hand_embd)[0]
        query_seq_len = tf.shape(query_joint_embd)[1]

        self.decoder.reset_cache(batch_size)

        state = None
        joint_embds = tf.TensorArray(tf.float32, size=query_seq_len, element_shape=[None, batch_size, self.cfg.embd_dim])
        for i in tf.range(query_seq_len):
            state, joint_embd = self.decoder(
                kv_embd=hand_embd,
                q_embd=query_joint_embd[:, i:i+1, :],
                kv_idxs=None,
                q_idxs=None,
                mask_type="none",
                state=state,
                write_state=True,
            )
            joint_embd = tf.transpose(joint_embd, [1, 0, 2])
            joint_embds = joint_embds.write(i, joint_embd)

        new_joint_embds = joint_embds.concat()
        new_joint_embds = tf.transpose(new_joint_embds, [1, 0, 2])
        # def while_cond(i, _n, _joint_embd, _all_joint_embd):
        #     return tf.less(i, query_seq_len)

        # @tf_scope
        # def while_body(i, n, prev_joint_embd, all_joint_embd):
        #     joint_query = query_joint_embd[:, i:i+1, :]

        #     n = n + tf.shape(prev_joint_embd)[1]
        #     new_joint_embd = self.decoder(
        #         kv_embd=prev_joint_embd,
        #         q_embd=joint_query,
        #         state=True,
        #         write_state=True,
        #         mask_type="none"
        #     )

        #     all_joint_embd = tf.concat(
        #         [all_joint_embd, new_joint_embd], axis=1)
        #     return [i+1, n, new_joint_embd, all_joint_embd]

        # _i, _n, _joint_embds, new_joint_embds = tf.while_loop(
        #     cond=while_cond,
        #     body=while_body,
        #     loop_vars=[
        #         tf.constant(0),
        #         tf.constant(0),
        #         tf.concat([hand_embd, cond_joint_embd], axis=1),
        #         tf.zeros([batch_size, 0, self.cfg.embd_dim]),
        #     ],
        #     shape_invariants=[
        #         tf.TensorShape([]),
        #         tf.TensorShape([]),
        #         tf.TensorShape([None, None, self.cfg.embd_dim]),
        #         tf.TensorShape([None, None, self.cfg.embd_dim]),
        #     ],
        #     maximum_iterations=17,
        # )

        # new_joint_embds = tf.ensure_shape(new_joint_embds, [batch_size, query_joint_embd.shape[1], self.cfg.embd_dim])

        return state, new_joint_embds


class EulerAngleDecoder(tf.Module):
    """
    Decodes euler angles into latent vectors.
    """
    @tf_scope

    def __init__(self, cfg, name="euler_decoder"):
        super().__init__(name=name)
        self.decoder = IRMQALayer(cfg, rpe=None)

    @tf_scope
    def __call__(self, joint_embd, euler_query_embd):

        batch_size = tf.shape(joint_embd)[0]
        self.decoder.reset_cache(batch_size)

        # tf.ensure_shape(self.decoder.irmqa.v_cache, [0, batch_size, 102])

        query = euler_query_embd[:, 0:1]
        euler_a = self.decoder(
            kv_embd=joint_embd,
            q_embd=query,
            state=True,
            write_state=True,
            mask_type="none",
        )

        # tf.ensure_shape(self.decoder.irmqa.v_cache, [1, batch_size, 102])

        query = euler_query_embd[:, 1:2]
        euler_b = self.decoder(
            kv_embd=euler_a,
            q_embd=query,
            state=True,
            write_state=True,
            mask_type="none",
        )

        # tf.ensure_shape(self.decoder.irmqa.v_cache, [2, batch_size, 102])

        query = euler_query_embd[:, 2:3]
        euler_c = self.decoder(
            kv_embd=euler_b,
            q_embd=query,
            state=True,
            write_state=False,
            mask_type="none",
        )

        # tf.ensure_shape(self.decoder.irmqa.v_cache, [3, batch_size, 102])

        return tf.concat([euler_a, euler_b, euler_c], 1)


class HierarchicalHandPredictor(Model):

    @tf_scope
    def __init__(self, cfg, decoder, name="hhp"):
        super().__init__(name=name)

        self.cfg = cfg

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
            cfg.embd_dim, name=f'frame_angle_embd')
        self.joint_angles_embd = layers.Dense(
            cfg.embd_dim, name=f'single_angle_embd')

        self.hand_encoder = IRMQAEncoder(cfg | cfg.hand_encoder, rpe=self.rpe)
        self.hand_decoder = IRMQALayer(cfg | cfg.hand_decoder, rpe=self.rpe, name="hand_dec")
        self.joint_decoder = MultiJointDecoder(cfg | cfg.joint_decoder)
        self.dof_decoder = EulerAngleDecoder(cfg | cfg.dof_decoder)

        self.dof_to_params = decoder
    
    def get_config(self):
        return self.cfg

    @tf_scope
    def reset_cache(self, batch_size):
        self.hand_encoder.reset_cache(batch_size)
        self.hand_decoder.reset_cache(batch_size)
    
    def compute_output_shape(self, input_shapes):
        return [
            input_shapes["query_j_idxs"][0],
            input_shapes["query_j_idxs"][1] * input_shapes["query_j_idxs"][2] * self.cfg.n_dof_per_joint,
            self.cfg.embd_dim
        ]

    def __call__(self, inputs, training, state=None, write_state=False):

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

        # hand encoder
        encoded_hand_embd = self.hand_encoder(
            embd=hand_embd,
            idxs=cond_frame_idxs,
            mask_type=self.mask_type,
            state=state,
            write_state=write_state,
        )

        # hand decoder
        query_frame_idxs = query_fh_idxs[..., 0]
        query_hand_idxs = query_fh_idxs[..., 1]
        hand_query_embd = self.hand_embd(query_hand_idxs)
        new_hand_embd = self.hand_decoder(
            kv_embd=encoded_hand_embd,
            kv_idxs=cond_frame_idxs,
            q_embd=hand_query_embd,
            q_idxs=query_frame_idxs,
            mask_type=self.mask_type,
            state=state,
            write_state=write_state,
        )

        # joint decoder
        joint_cond_embd = self.joint_angles_embd(
            cond_joint_vecs) + self.joint_embd(cond_j_idxs)
        joint_cond_embd = ein.rearrange(
            joint_cond_embd, 'b fh j e -> (b fh) j e')
        joint_query_embd = self.joint_embd(query_j_idxs)
        joint_query_embd = ein.rearrange(
            joint_query_embd, 'b fh j e -> (b fh) j e')
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

        output_params = self.dof_to_params(output_embd)

        return output_params


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
def hierarchical_batched_random_chunk(cfg, x, seed=None):

    @tf_scope
    def get_chunk(angles):
        angles = angles

        angles = ein.rearrange(
            angles, 'f h (j d) -> f h j d', j=cfg.n_joints_per_hand, d=cfg.n_dof_per_joint)
        n_frames = tf.shape(angles)[0]
        fh_idxs = utils.multidim_indices([n_frames, cfg.n_hands], flatten=True)
        if cfg.contiguous:
            i = tf.random.uniform(
                [], minval=0, maxval=n_frames-cfg.n_hand_vecs, seed=seed, dtype=tf.int32)
            idxs = tf.concat([fh_idxs[i:i+cfg.n_hand_vecs//2],
                             fh_idxs[i+1:i+cfg.n_hand_vecs//2+1]], axis=0)
        else:
            idxs = random_subset(fh_idxs, cfg.n_hand_vecs, seed=seed)
        hands = tf.gather_nd(angles, idxs)

        return hands, idxs

    hands, fh_idxs = tf.map_fn(
        get_chunk,
        x["angles"],
        fn_output_signature=(
            tf.TensorSpec(shape=[None, cfg.n_joints_per_hand,
                          cfg.n_dof_per_joint], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
        ),
    )
    if cfg.contiguous:
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

    cond_joint_vecs = tf.gather(target_hands, cond_j_idxs, batch_dims=2)
    target_joint_vecs = tf.gather(target_hands, query_j_idxs, batch_dims=2)
    target_joint_vecs = ein.rearrange(target_joint_vecs, 'b fh j d -> b (fh j d)')
    return (
        {
            "cond_hand_vecs": cond_hand_vecs,
            "cond_fh_idxs": cond_fh_idxs,
            "query_fh_idxs": query_fh_idxs,
            "cond_joint_vecs": cond_joint_vecs,
            "cond_j_idxs": cond_j_idxs,
            "query_j_idxs": query_j_idxs,
        },
        target_joint_vecs,
    )


def dream_dataset(cfg, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset):

    # train input isn't always frame aligned
    train_dataset = train_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=cfg.batch_size))
    train_dataset = train_dataset.map(
        lambda x: hierarchical_batched_random_chunk(cfg, x))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # test input is always frame-aligned
    # take fixed size chunks from the tensor at random frame indices
    # N = 100 repeats * 12 examples = 1200 test examples
    test_dataset = test_dataset.repeat(100)
    test_dataset = test_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=cfg.test_batch_size))
    test_dataset = test_dataset.map(
        lambda x: hierarchical_batched_random_chunk(cfg, x, seed=1234))

    # val input is always frame-aligned
    # take fixed size chunks from the tensor at random frame indices
    # N = 100 repeats * 12 examples = 1200 test examples
    val_dataset = val_dataset.repeat(10)
    val_dataset = val_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=cfg.batch_size))
    val_dataset = val_dataset.map(
        lambda x: hierarchical_batched_random_chunk(cfg, x, seed=1234))

    return train_dataset, test_dataset, val_dataset


def train(cfg, run_name):

    cfg = cfg | cfg.dream

    pprint(dict(cfg))

    d_train, d_test, d_val = data_tf.tf_dataset(cfg, dream_dataset)

    vmf_loss, vmf_dist, vmf_mean, vmf_sample, vmf_decoder = decoders.von_mises_fisher(
        cfg, name="vmf")

    inp, tar = next(iter(d_train))

    inp_shape = {k: v.shape for k, v in inp.items()}
    print()
    pprint(inp_shape)
    inp_dtype = {k: v.dtype for k, v in inp.items()}
    pprint(inp_dtype)
    print()

    # inputs = {
    #     "cond_hand_vecs": Input(shape=[None, cfg.n_joints_per_hand * cfg.n_dof_per_joint], dtype=tf.float32, name="cond_hand_vecs"),
    #     "cond_fh_idxs": Input(shape=[None, 2], dtype=tf.int32, name="cond_fh_idxs"),
    #     "query_fh_idxs": Input(shape=[None, 2], dtype=tf.int32, name="query_fh_idxs"),
    #     "cond_joint_vecs": Input(shape=[None, None, cfg.n_dof_per_joint], dtype=tf.float32, name="cond_joint_vecs"),
    #     "cond_j_idxs": Input(shape=[None, None], dtype=tf.int32, name="cond_j_idxs"),
    #     "query_j_idxs": Input(shape=[None, None], dtype=tf.int32, name="query_j_idxs"),
    # }
    model = HierarchicalHandPredictor(cfg | cfg.model, vmf_decoder)
    optimizer = tf.keras.optimizers.Adam(cfg.adam.lr)
    train_loop(model, vmf_loss, optimizer, cfg.steps, cfg.steps_per_epoch, d_train, d_test, d_val)

def train_loop(model, loss_fn, optimizer, n_steps, n_steps_per_epoch, d_train, d_test, d_val):
    
    n_chunk_steps = 10
    
    def chunk_fn():
        """
        Run a small number of steps in a loop to improve
        execution speed.
        """
        for _ in range(n_chunk_steps):
            inputs, targets = next(iter(d_train))
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                loss = loss_fn(targets, outputs)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        return loss

    i = 0
    while i < n_steps:
        if i % n_steps_per_epoch == 0:
            epoch = i // n_steps_per_epoch
            print(f"epoch {epoch}")
        loss = chunk_fn()
        i += n_chunk_steps
        print(f"step: {i}, loss: {loss}")
        
