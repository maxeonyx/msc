import einops as ein
from einops.layers.keras import EinMix
import tensorflow as tf
from ml import data_tf, encoders, utils
from tensorflow.python import keras
from tensorflow.python.keras import Input, Model, layers

import deberta

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


class RelativePositionEmbedding(layers.Layer):

    def __init__(self, max_N, D,  allow_negative=False, trainable=True, name="rel_embd", dtype=None, dynamic=False, **kwargs):
        """
        If the computation only ever uses
        relative attention in one direction (eg. incremental multi-query layers with causal masking),
        then allow_negative can be set to false to halve memory use.
        """
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
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


class RelativePositionProjectionLookup(layers.Layer):

    # def __init__(self, projection, relative_pos_embedding):
    #     self.rpe = relative_pos_embedding
    #     # seq dim should be first axis for the tf.gather below
    #     self.projection = projection
    #     self.d
    #     self.cache = None

    def __init__(self, trainable=True, name="rel_projection", dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
    
    def call(self, relative_pos):
        if self.cache is None:
            rel_embd = self.rpe.get_all_embeddings()
            self.cache = self.projection(rel_embd)
        indices = self.rpe.get_indices(relative_pos)
        rel_proj = tf.gather(self.cache, indices, axis=0)
        
        return rel_proj


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


    def __init__(self, QK, V, D, H, rpe, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.supports_masking = True

        self.QK = QK
        self.V = V
        self.D = D
        self.H = H

        self.wv = EinMix('b m d -> m b v', weight_shape='d v', d=self.D, v=self.V)

        self.wqc = EinMix('b m d -> b h m qk', weight_shape='d h qk', d=self.D, h=self.H, qk=self.QK)
        # k & v have m dimension first so they can be put into TensorArray
        self.wkc = EinMix('b m d -> m b qk', weight_shape='d qk', d=self.D, qk=self.QK)

        if rpe is not None:
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


    def call(self, z_embd, rel_pos, q_embd=None):

        # MZ is the number of new inputs this iteration, the results of which will be cached.
        #    Might be M=N, or might be M=1.
        # MQ is the number of queries this iteration.
        # N is the total number of inputs so far, including any new inputs. (so N >= MZ)

        # z_embd is [B, MZ, D]
        # rel_pos is [B, MQ, N] (int32)
        # q_embd is [B, MQ, D]. If q_embd is None, it is assumed to be z_embd, so 
        #                       MQ == MZ
        # mask is [N, N]
        # returns [B, MQ, D]
        # and increases N_t to N_{t+1}= N_t + MZ

        if q_embd is None:
            q_embd = z_embd

        i = self.k_cache.size()

        v = self.wv(z_embd)
        # add MZ new values to the cache
        self.v_cache.write(i, v)
        v = self.v_cache.concat()

        rel_pos = tf.ensure_shape(rel_pos, [q_embd.shape[0], q_embd.shape[1], v.shape[0]])

        # c2c
        qc = self.wqc(q_embd)
        kc = self.wkc(z_embd)

        # add MZ new keys to the cache
        self.kc_cache.write(i, kc)
        kc = self.kc_cache.concat()

        attn_logits = tf.einsum('bhmk,bnk->bhmn', qc, kc)

        # if using relative positions, use disentangled attention
        if rel_pos is not None:
            n_attn_types = 3

            # p2c
            qp = self.wqp(rel_pos)
            # kc
            # h second last in this one because of embedding lookup
            attn_logits += tf.einsum('bmnhk,bnk->bhmn', qp, kc)

            # c2p
            # qc
            kp = self.wkp(rel_pos)
            attn_logits += tf.einsum('bhmk,bmnk->bhmn', qc, kp)
        else: 
            n_attn_types = 1
        
        scale = 1. / tf.sqrt(self.D * n_attn_types)
        attn_logits *= scale
        attn_logits = self.dropout(attn_logits)
        attn_weights = self.softmax(attn_logits)
    
        v_h = tf.einsum('bhmn,bnv->bhmv', attn_weights, v) # value-per-head
        v_o = self.wo(v_h) # project to output dimension
        return v_o

class IRMQALayer(layers.Layer):
    """
    MQA, Intermediate and Output parts.
    """

    def __init__(self, cfg, rpe, trainable=True, name="irmqa", dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.supports_masking = True

        self.irmqa = IRMQA(
            QK=cfg.qk_dim,
            V=cfg.v_dim,
            D=cfg.d_dim,
            H=cfg.n_heads,
            rpe=rpe,
            name=f"{name}_attn"
        )
        self.intermediate = deberta.TFDebertaIntermediate(
            intermediate_size=cfg.intermediate_size,
            initializer_range=cfg.initializer_range,
            hidden_act=cfg.hidden_act,
            name=f"{name}_intermediate"
        )
        self.output = deberta.TFDebertaOutput(
            initializer_range=cfg.initializer_range,
            hidden_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.hidden_dropout_prob,
            layer_norm_eps=cfg.layer_norm_eps,
            name=f"{name}_output"
        )
    
    def call(self, embd, rel_pos, mask, training, q_embd=None):

        embd = self.irmqa(embd, rel_pos, mask, training, q_embd=q_embd)
        embd = self.intermediate(embd)
        embd = self.output(embd)

        return embd

class HandEncoder(layers.Layer):
    """
    Encoder for the previous
    """

    def __init__(self, cfg, name="hand_enc", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

        self.n_layers = cfg.n_layers
        self.embd_dim = cfg.embd_dim
        
        self.irmqa_layers = [
            IRMQALayer(cfg, name=f"{name}_irmqa_{i}") for i in range(self.n_layers)
        ]
    
    def call(self, embd, rel_pos):

        batch_size = tf.shape(embd)[0]
        mask = encoders.all_attention_mask(batch_size, embd.shape[1], embd.shape[2])
        for layer in self.irmqa_layers:
            embd = layer(embd, rel_pos, mask)
        
        return embd

class HandDecoder(layers.Layer):
    """
    Decodes target hands, specified by embeddings and relative positions.
    """

    def __init__(self, cfg, rpe, name="hand_dec", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.hand_query_embedding = layers.Embedding(2, cfg.embd_dim, name=f"{name}/query_embedding")
        self.decoder = IRMQALayer(cfg, rpe, name=f"{name}/irmqa")
    
    def call(self, kv_embd, q_hand_idxs, rel_pos):

        batch_size = tf.shape(embd)[0]
        mask = encoders.all_attention_mask(batch_size, q_hand_idxs.shape[1], kv_embd.shape[2])

        q_embds = self.hand_query_embedding(q_hand_idxs)

        for layer in self.irmqa_layers:
            embd = layer(embd, rel_pos, mask)
        
        return embd

def random_joint_order():
    """
    Produce a random selection of joint indices, such that they
    still obey the hierarchy of the hand skeleton. i.e. the
    wrist pos must be produced first, then the joints of each
    finger must be produced in order (although they might be
    interspersed.)

    Uses tensorflow only in order to run in data pipeline.
    """
    use_wrist = tf.random.categorical(tf.math.log([0.1, 0.9]), 1, tf.int32)[0]
    
    joint_idxs = tf.constant([], dtype=tf.int32)
    if tf.equal(use_wrist, 0):
        return joint_idxs

    wrist_idx = tf.constant([0])
    thumb_idxs = tf.constant([1, 2, 3, 4])
    index_idxs = tf.constant([5, 6, 7])
    middle_idxs = tf.constant([8, 9, 10])
    ring_idxs = tf.constant([11, 12, 13])
    pinky_idxs = tf.constant([14, 15, 16])
    
    joint_idxs = tf.concat([joint_idxs, wrist_idx], 0)
    
    n_thumb_joints = tf.random.categorical(tf.math.log([0.2, 0.2, 0.2, 0.2, 0.2]), 1)[0]
    n_index_joints = tf.random.categorical(tf.math.log([0.25, 0.25, 0.25, 0.25]), 1)[0]
    n_middle_joints = tf.random.categorical(tf.math.log([0.25, 0.25, 0.25, 0.25]), 1)[0]
    n_ring_joints = tf.random.categorical(tf.math.log([0.25, 0.25, 0.25, 0.25]), 1)[0]
    n_pinky_joints = tf.random.categorical(tf.math.log([0.25, 0.25, 0.25, 0.25]), 1)[0]

    joint_order = tf.concat([
        tf.repeat([0], n_thumb_joints),
        tf.repeat([1], n_index_joints),
        tf.repeat([2], n_middle_joints),
        tf.repeat([3], n_ring_joints),
        tf.repeat([4], n_pinky_joints),
    ])
    joint_order = tf.random.shuffle(joint_order)

    i_thumb = 0
    i_index = 0
    i_middle = 0
    i_ring = 0
    i_pinky = 0
    for i in joint_order:
        if tf.equal(i, 0):
            joint_idxs = tf.concat([joint_idxs, thumb_idxs[i_thumb]], 0)
            i_thumb += 1
        elif tf.equal(i, 1):
            joint_idxs = tf.concat([joint_idxs, index_idxs[i_index]], 0)
            i_index += 1
        elif tf.equal(i, 2):
            joint_idxs = tf.concat([joint_idxs, middle_idxs[i_middle]], 0)
            i_middle += 1
        elif tf.equal(i, 3):
            joint_idxs = tf.concat([joint_idxs, ring_idxs[i_ring]], 0)
            i_ring += 1
        elif tf.equal(i, 4):
            joint_idxs = tf.concat([joint_idxs, pinky_idxs[i_pinky]], 0)
            i_pinky += 1
    
    return joint_idxs

class MultiJointDecoder(layers.Layer):
    """
    Decodes all joints of a hand in any valid topological order.
    """

    def __init__(self, cfg, trainable=True, name="joint_decoder", dtype=None, dynamic=False, **kwargs):
        
        self.cfg = cfg

        self.joint_query_embeddings = layers.Embedding(17, cfg.embd_dim, name=f"{name}/query_embeddings")

        self.decoder = IRMQALayer(cfg.layer, rpe=None, name=f"{name}/irmqa")
        
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    
    def call(self, hand_embd, joint_order, training, *args, **kwargs):

        i = tf.constant(0)

        batch_size = tf.shape(hand_embd)[0]
        joint_embds = tf.zeros_like([batch_size, 0, self.cfg.embd_dim])

        while_cond = lambda i: tf.less(i, tf.shape(joint_order)[0])
        def while_body(i, joint_embds):
            joint_idx = joint_order[i]
            joint_query = self.joint_query_embeddings(joint_idx)
            
            mask = encoders.all_attention_mask(batch_size, 1, tf.shape(joint_embds)[1])
            new_joint_embd = self.decoder(joint_embds, q_embd=joint_query, rel_pos=None, mask=mask, training=training)

            joint_embds = tf.concat([joint_embds, new_joint_embd], 1)
            i += 1
            return i, joint_embds

        _i, joint_embds = tf.while_loop(
            while_cond,
            while_body,
            [i, joint_embds],
            shape_invariants=[
                tf.shape(i),
                tf.TensorShape([batch_size, None, self.cfg.embd_dim]),
            ],
            maximum_iterations=17,
        )

        return joint_embds


class EulerAngleDecoder(layers.Layer):
    """
    Decodes euler angles into latent vectors.
    """
    
    def __init__(self, cfg, trainable=True, name="euler_decoder", dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        
        self.euler_query_embeddings = layers.Embedding(3, cfg.embd_dim, name="euler_query_embeddings")

        self.decoder = IRMQALayer(cfg.layer, rpe=None, name=f"{name}_irmqa")

    
    def call(self, joint_embds, *args, **kwargs):

        batch_size = tf.shape(joint_embds)[0]

        kv_embd = joint_embds

        mask = encoders.all_attention_mask(batch_size, 1, kv_embd.shape[1], tf.bool)
        euler_a = self.decoder(z_embd=joint_embds, rel_pos=None, mask=mask)
        kv_embd = tf.concat([kv_embd, euler_a], 1)

        mask = encoders.all_attention_mask(batch_size, 1, kv_embd.shape[1], tf.bool)
        euler_b = self.decoder(z_embd=joint_embds, rel_pos=None, mask=mask)
        kv_embd = tf.concat([kv_embd, euler_b], 1)

        mask = encoders.all_attention_mask(batch_size, 1, kv_embd.shape[1], tf.bool)
        euler_c = self.decoder(z_embd=joint_embds, rel_pos=None, mask=mask)

        return tf.concat([euler_a, euler_b, euler_c], 1)

class HierarchicalHandPredictor(layers.Layer):

    def __init__(self, cfg, trainable=True, name="hhp", dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.rpe = RelativePositionEmbedding(
            # maximum number of inputs.
            # defines the size of the single "rel_embd" cache
            # and n_layers "rel_proj" caches
            max_N=cfg.max_N,
            D=cfg.embd_dim,
            name=f"{name}/rel_embd"
        )

        # token to represent 'empty' conditioning info
        self.empty_token = layers.Embedding(1, cfg.embd_dim, name=f'{name}/begin_token_embd')
        self.hand_embd = layers.Embedding(cfg.n_hands, cfg.embd_dim, name=f'{name}/hand_embd')
        self.joint_embd = layers.Embedding(cfg.n_joints, cfg.embd_dim, name=f'{name}/joint_embd')
        self.euler_embd = layers.Embedding(3, cfg.embd_dim, name=f'{name}/euler_embd')

        self.frame_angles_embd = layers.Dense(cfg.embd_dim, name=f'{name}/frame_angle_embd')
        self.single_angle_embd = layers.Dense(cfg.embd_dim, name=f'{name}/single_angle_embd')

        self.hand_encoder = HandEncoder(cfg.hand_encoder, name=f"{name}/hand_encoder")
        self.hand_decoder = HandDecoder(cfg.hand_decoder, name=f"{name}/hand_decoder")
        self.joint_decoder = MultiJointDecoder(cfg.joint_decoder, name=f"{name}/joint_decoder")
        self.dof_decoder = EulerAngleDecoder(cfg.dof_decoder, name=f"{name}/dof_decoder")

    def reset_cache(self):


    
    def call(self, inputs, *args, **kwargs):

        full_frames = inputs["full_frames"]
        full_frame_idxs = inputs["full_frame_idxs"]
        partial_frame_idxs = inputs["partial_frame_idxs"]
        partial_frame_idxs = inputs["partial_input_idxs"]
        partial_inputs = inputs["partial_inputs"]

        if tf.equal(tf.shape(full_frames)[1], 0):
            # if there's no conditioning info, return a batch of the "empty" token
            frame_embd = self.empty_token(tf.zeros([tf.shape(full_frames)[0], 1], dtype=tf.int32))
        else:
            frame_idxs = full_frame_idxs[..., 0]
            hand_idxs = full_frame_idxs[..., 1]
            frame_embd = self.frame_angles_embd(full_frames) + self.hand_embd(hand_idxs)
        
        full_rel_pos = full_frame_idxs[..., None, :] - full_frame_idxs[..., :, None]
        frame_embd = self.hand_encoder(frame_embd, full_rel_pos)

        hand_query_embd = 


def random_subset(options, n, seed=None):
    options = tf.random.shuffle(options, seed=seed)
    return options[:n]

def weighted_random_n(n_max, D=2., B=10., seed=None):
    """
    Picks random indices from [0, n_max) with skewed distribution
    towards the lower numbers.
    p(index i) = 1/D * p(index Di) = 1/D**2 * p(index D**2i)
    The probability of each index decreases by D every power of b
    """
    logb1p = lambda b, x: tf.math.log1p(x)/tf.math.log(b)
    # probabilities
    # probabilities = l1_norm(powf(D, -logb1p(B, tf.range(n_max))))
    # log probabilities base e
    log_probabilities = -logb1p(B, tf.range(n_max))*tf.math.log(D)

    return tf.random.categorical(log_probabilities[None, :], 1, seed=seed)[0, 0]

def hierarchical_batched_random_chunk(cfg, x, seed=None):

    angles = x["angles"]
    angles = tf.ensure_shape(angles, [None, cfg.n_hands, cfg.n_dof])

    n_frames = tf.shape(angles)[0]
    n_hands = tf.shape(angles)[1]
    n_joints = tf.shape(angles)[2]
    n_dof = tf.shape(angles)[3]

    batch_size = cfg.batch_size

    sincos = ein.rearrange([tf.sin(angles), tf.cos(angles)], 's f h d -> (f h) j d s', d=3, j=17)

    frame_idxs = tf.map_fn(
        lambda: random_subset(tf.range(n_frames), cfg.n_chunk_frames, seed=seed),
        tf.range(batch_size)
    )
    n_full_frames = weighted_random_n(cfg.n_chunk_frames, D=100., B=cfg.n_chunk_frames, seed=seed) # 100 times more likely to have 0 than 50
    full_frame_idxs = frame_idxs[:, :n_full_frames]
    partial_frame_idxs = frame_idxs[:, n_full_frames:]

    full_frames = tf.gather(sincos, full_frame_idxs)
    
    n_partial_frame_inputs = weighted_random_n(n_hands*n_dof, D=10., B=n_hands*n_dof, seed=seed)

    batch_frame_idxs = utils.multidim_indices_of(partial_frame_idxs, flatten=False)
    flat_b_f_i = ein.rearrange(batch_frame_idxs, 'b f i -> (b f) i')
    hand_dof_idxs = utils.multidim_indices([n_hands, n_dof], flatten=True)
    partial_input_idxs = tf.map_fn(
        lambda i_b, i_f: random_subset(hand_dof_idxs, n_partial_frame_inputs, seed=seed),
        flat_b_f_i,
    )
    # add frame idxs to partial input idxs
    partial_frame_input_idxs = tf.concat([partial_frame_idxs, partial_input_idxs], axis=-1)
    partial_inputs = tf.gather_nd(sincos, partial_frame_input_idxs)

    return {
        "full_frames": full_frames,
        "full_frame_idxs": full_frame_idxs,
        "partial_inputs": partial_inputs,
        "partial_frame_idxs": partial_frame_idxs,
        "partial_input_idxs": partial_input_idxs,
    }

def dream_dataset(cfg, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset):
    
    # train input isn't always frame aligned
    train_dataset.map(lambda x: hierarchical_batched_random_chunk(cfg, x))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # test input is always frame-aligned
    # take fixed size chunks from the tensor at random frame indices
    test_dataset = test_dataset.repeat(100) # N = 100 repeats * 12 examples = 1200 test examples
    train_dataset.map(lambda x: hierarchical_batched_random_chunk(cfg, x, seed=1234))

    return train_dataset, test_dataset


def train(cfg):

    d_train, d_test = data_tf.tf_dataset(cfg, dream_dataset)

    

    # 
    # model
    #  - embedder
    #  - encoder
    #  - decoder
    # 
    # 
