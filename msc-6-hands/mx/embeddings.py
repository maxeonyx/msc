from __future__ import annotations
from enum import Enum

from mx.prelude import *
from mx.models import DecoderOnlyTransformer
from mx.pipeline import DiscreteEmbedding_TaskConfig, Embedding_TaskConfig, MxEmbedding, FloatEmbedding_TaskConfig, Model_EmbeddingConfig, MxModel


@export
class PrependToken(u.MxLayer):
    """
    Prepend a sentinel embedding to the beginning of the sequence.

    >>> n_embd = 1
    >>> initializer = keras.initializers.Constant(0.)
    >>> prepend_token = PrependToken(embeddings_initializer=initializer)
    >>> embd = tf.ones([1, 1, n_embd])
    >>> embd = prepend_token(embd)
    >>> embd.shape
    TensorShape([1, 2, 1])
    >>> print(embd.numpy())
    [[[0.]
      [1.]]]
    >>> n_embd = 4
    >>> prepend_token = PrependToken(embeddings_initializer=initializer)
    >>> embd = tf.ones([1, 1, n_embd])
    >>> embd = prepend_token(embd)
    >>> embd.shape
    TensorShape([1, 2, 4])
    >>> print(embd.numpy())
    [[[0. 0. 0. 0.]
      [1. 1. 1. 1.]]]
    """

    def __init__(
        self,
        name="prepend_token",
        desc="Prepend Token",
        embeddings_initializer=None,
        **kwargs,
    ):
        super().__init__(name, desc, **kwargs)
        self.embeddings_initializer = embeddings_initializer


    def build(self, input_shape):
        n_embd = input_shape[-1]
        initializer = self.embeddings_initializer or tf.keras.initializers.TruncatedNormal(stddev=1/sqrt(n_embd))
        self.token_embedding = layers.Embedding(1, n_embd, embeddings_initializer=initializer, name=f"{self.name}/token_embd")


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])

    def call(self, inputs):
        embd = inputs

        batch_size = shape(embd)[0]
        tokens = tf.tile(
            tf.constant(0, tf.int32)[None, None],
            [batch_size, 1],
        )
        token_embd = self.token_embedding(tokens)
        embd = tf.concat([token_embd, embd], axis=1) # concat along first sequence dim
        return embd

@export
class PosEnc(u.MxLayer):
    """
    Sinusoidal positional encoding.

    >>> pos_enc = PosEnc(4)
    >>> idxs = { 'idxs': u.multidim_indices([10]) }
    >>> pos_enc(idxs).shape
    TensorShape([10, 4])
    >>> pos_enc = PosEnc(4)
    >>> idxs_with_batch = { 'idxs': ein.repeat(u.multidim_indices([10]), '... -> b ...', b=3) }
    >>> pos_enc(idxs_with_batch).shape
    TensorShape([3, 10, 4])
    >>> pos_enc = PosEnc(4, n_seq_dims=2, max_wavelengths=[10000, 10000])
    >>> idxs = { 'idxs': u.multidim_indices([10, 10]) }
    >>> pos_enc(idxs).shape
    TensorShape([100, 4])
    """

    def __init__(
        self,
        n_embd,
        n_seq_dims=1,
        max_wavelengths=None,
        name="pos_enc",
        desc=None,
        **kwargs
    ):
        super().__init__(name=name, desc=desc, **kwargs)

        assert n_embd % (n_seq_dims*2) == 0, f"embd_dim must be divisible by 2 and by n_seq_dims to use positional encoding, got embd_dim={n_embd}, n_seq_dims={n_seq_dims}"

        self.n_embd = n_embd
        self.n_seq_dims = n_seq_dims
        if max_wavelengths is None:
            self.max_wavelengths = [10000] * n_seq_dims
        elif n_seq_dims == 1 and isinstance(max_wavelengths, int):
            self.max_wavelengths = [max_wavelengths]
        else:
            assert len(max_wavelengths) == n_seq_dims, f"max_wavelengths must have length n_seq_dims, got {len(max_wavelengths)} and {n_seq_dims}"
            self.max_wavelengths = max_wavelengths

    # based from the keras source code
    # https://github.com/keras-team/keras-nlp/blob/v0.3.0/keras_nlp/layers/sine_position_encoding.py#L21

    @u.tf_scope
    def pos_enc(self, idxs, n, w):
        position = tf.cast(idxs, u.dtype())
        min_freq = 1. / w
        timescales = tf.pow(
            min_freq,
            tf.range(n, dtype=tf.float32) / n
        )
        timescales = tf.cast(timescales, u.dtype())
        position = ein.rearrange(position, '... seq -> ... seq ()')
        timescales = ein.rearrange(timescales, '... embd -> ... () embd')
        angles = position * timescales
        # even indices are sine, odd are cosine
        cos_mask = tf.cast(tf.range(n) % 2, u.dtype())
        sin_mask = 1 - cos_mask
        # embedding shape is [seq_length, hidden_size]
        positional_encodings = (
            tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
        )

        # scale norm. because we use sin/cos we scale by 1/sqrt(D/2) instead of 1/sqrt(D)
        positional_encodings *= tf.cast(tf.math.sqrt(tf.cast(n // 2, u.dtype())), u.dtype())

        return positional_encodings

    @u.tf_scope
    def call(self, inputs):
        idxs = inputs["idxs"]

        positional_encodings = [
            self.pos_enc(
                idxs[..., i],
                self.n_embd // self.n_seq_dims,
                self.max_wavelengths[i]
            )
            for i in range(self.n_seq_dims)
        ]

        idx_embd = tf.concat(positional_encodings, axis=-1)

        return idx_embd


@export
@dataclass
class SeqEmbd_TaskConfig(FloatEmbedding_TaskConfig):
    sequence_length: int
    """
    Max length of the sequence to be embedded.
    Max value among seq_idxs.
    """

    chunk_length: int

@export
class AngleCodebook(MxEmbedding):
    """
    Simple embedding for transformer regression over angles.

    Embeds angles as unit vectors. Creates `n_repeats` copies of them, rotated
    evenly around one quarter of the unit circle.

    Embeds positions with a codebook.
    """

    def __init__(
        self,
        n_embd: int,
        n_repeats: int,
        name="angleembd",
        desc="Angles: `n_repeats` rotations -> linear layer. Positions: Abs position codebook.",
    ) -> None:
        super().__init__(
            n_embd=n_embd,
            name=name,
            desc=desc,
        )
        self.n_repeats = n_repeats
        "Number of additional rotations of each angle to be added to the embedding."

        self.task_config_type: Type[SeqEmbd_TaskConfig] = SeqEmbd_TaskConfig
        self.task_cfg: SeqEmbd_TaskConfig | None = None

    def configure(self, model: MxModel):
        if isinstance(model, DecoderOnlyTransformer):
            model.recieve_embd_config(model.embd_cfg_type(
                n_embd=self.n_embd,
            ))
        else:
            raise NotImplementedError(f"Embedding {type_name(self)} does not support Model {type_name(model)}. If using autoreload in IPython, try restarting the interpreter.")

    def make_embedder(self) -> Model:
        "Creats the keras model for the embedding."

        assert self.n_embd % 2 == 0, f"n_embd must be divisible by 2 to use angle embedding, got n_embd={self.n_embd}"
        assert self.task_cfg is not None, "Must call task.configure(embedding) before embedding.make_embedder()."

        pos_embedder = layers.Embedding(self.task_cfg.sequence_length, self.n_embd, name="pos_embedder")
        dense_out = layers.Dense(self.n_embd, name="embd")

        import mx.layers as mxl
        prepend_begin_token = prepend_token(
            token=tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):

            ## make angle embeddings
            angles = inputs["context/vals"]

            angles = tf.cast(angles, u.dtype())

            angle_embd = ein.repeat(
                angles,
                "... seq feat -> ... seq (feat rep)",
                rep=self.n_repeats,
            )
            angle_embd /= tf.sqrt(tf.cast(shape(angle_embd)[-1], u.dtype()))

            ## make position embeddings
            pos_idxs = inputs["context/inp_idxs"]
            pos_embd = pos_embedder(pos_idxs)

            return prepend_begin_token(angle_embd + pos_embd)

        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="context/vals"),
            Input([None, 1],                          dtype=tf.int32,  name="context/inp_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        ), inputs


@export
class AngleCodebookMultidim(MxEmbedding):
    """
    Simple embedding for transformer regression over angles.

    Embeds angles as unit vectors. Creates `n_repeats` copies of them, rotated
    evenly around one quarter of the unit circle.

    Embeds positions with a codebook.
    """

    def __init__(
        self,
        n_embd: int,
        n_repeats: int,
        name="angleembd",
        desc="Angles: `n_repeats` rotations -> linear layer. Positions: Abs position codebook.",
    ) -> None:
        super().__init__(
            n_embd=n_embd,
            name=name,
            desc=desc,
        )
        self.n_repeats = n_repeats
        "Number of additional rotations of each angle to be added to the embedding."

        self.task_config_type: Type[CodebookMultidim_TaskConfig] = CodebookMultidim_TaskConfig
        self.task_cfg: CodebookMultidim_TaskConfig | None = None

    def configure(self, model: MxModel):
        if isinstance(model, DecoderOnlyTransformer):
            model.recieve_embd_config(model.embd_cfg_type(
                n_embd=self.n_embd,
            ))
        else:
            raise NotImplementedError(f"Embedding {type_name(self)} does not support Model {type_name(model)}. If using autoreload in IPython, try restarting the interpreter.")

    def make_embedder(self) -> Model:
        "Creats the keras model for the embedding."

        assert self.n_embd % 2 == 0, f"n_embd must be divisible by 2 to use angle embedding, got n_embd={self.n_embd}"
        assert self.task_cfg is not None, "Must call task.configure(embedding) before embedding.make_embedder()."

        pos_embedder = layers.Embedding(self.task_cfg.sequence_length, self.n_embd, name="pos_embedder")
        dense_out = layers.Dense(self.n_embd, name="embd")


        n_seq_dims = len(self.task_cfg.seq_dims)

        initializer = keras.initializers.TruncatedNormal(stddev=1./sqrt(self.n_embd))

        # embed using index within current window
        win_pos_embedder = layers.Embedding(self.task_cfg.seq_len, self.n_embd, name=f"win_pos_embd", embeddings_initializer=initializer)

        n_total = prod(self.task_cfg.seq_dims)

        abs_pos_embedder = layers.Embedding(n_total, self.n_embd, name=f"abs_pos_embd", embeddings_initializer=initializer)


        import mx.layers as mxl
        prepend_begin_token = prepend_token(
            token=tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):

            ## make angle embeddings
            angles = inputs["context/vals"]

            angles = tf.cast(angles, u.dtype())

            angle_embd = ein.repeat(
                angles,
                "... seq feat -> ... seq (feat rep)",
                rep=self.n_repeats,
            )
            angle_embd /= tf.sqrt(tf.cast(shape(angle_embd)[-1], u.dtype()))

            ## make position embeddings
            pos_idxs = inputs["context/inp_idxs"]
            pos_embd = pos_embedder(pos_idxs)

            return prepend_begin_token(angle_embd + pos_embd)

        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="context/vals"),
            Input([None, 1],                          dtype=tf.int32,  name="context/inp_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        ), inputs

@export
class AngleEmbd(u.MxLayer):
    """
    Embed angles as unit vectors and pass through a dense layer.

    >>> embd = AngleEmbd(n_embd=4)
    >>> embd(tf.constant([[[0.], [1.]]])).shape
    TensorShape([1, 3, 4])
    """

    def __init__(
        self,
        n_embd: int,
        embeddings_initializer = None,
        name="angleembd",

        desc=None,
    ):
        if desc is None:
            desc = "Angles: repeat -> unit vec -> linear layer."
        super().__init__(name=name, desc=desc)
        self.n_embd = n_embd
        self.embeddings_initializer = embeddings_initializer

    def build(self, input_shape):
        self.n_feat = input_shape[-1]
        self.n_repeats = max(1, self.n_embd // (self.n_feat*2))
        self.prepend_begin_val = PrependToken(embeddings_initializer=self.embeddings_initializer, name=f"{self.name}/prepend_begin_val")
        self.dense = layers.Dense(self.n_embd, name=f"{self.name}/dense")

    @tf.function
    @u.tf_scope
    def call(self, angles):

        angles = tf.stack(
            [
                tf.sin(angles),
                tf.cos(angles),
            ],
            axis=-1,
        )
        angles = ein.repeat(
            angles,
            "... feat sincos -> ... (feat rep sincos)",
            rep=self.n_repeats,
        )

        # flatten to "embd" dim
        angle_embd = self.dense(angles)

        angle_embd /= tf.sqrt(tf.cast(angles.shape[-1]//2, u.dtype()))

        return self.prepend_begin_val(angle_embd)


@export
class DecoderOnly_Angle_Codebook(u.MxLayer):
    """
    Embedding for decoder-only model. Vals with AngleEmbd, Positions with Codebook.

    >>> embd = DecoderOnly_Angle_Codebook(n_embd=4, seq_shape=[5])
    >>> inps = {
    ...     'ctx/inp/vals': tf.constant([[[0.], [1.]]]), # angles
    ...     'ctx/inp/idxs': tf.constant([[[0], [1]]]),  # positions
    ... }
    >>> embd(inps)['ctx/embd'].shape
    TensorShape([1, 3, 4])
    >>> embd = DecoderOnly_Angle_Codebook(n_embd=4, seq_shape=[5], include_target=True)
    >>> inps = {
    ...     'ctx/inp/vals': tf.constant([[[0.], [1.]]]), # angles
    ...     'ctx/inp/idxs': tf.constant([[[0], [1]]]),  # positions
    ...     'ctx/tar/idxs': tf.constant([[[0], [1], [2]]]),  # positions
    ... }
    >>> embd(inps)['ctx/embd'].shape
    TensorShape([1, 3, 4])
    """

    def __init__(
        self,
        n_embd: int,
        seq_shape: list[int],
        include_target: bool = False,
        name="acotrips",
        desc=None,
        **kwargs
    ):
        if desc is None:
            desc = "AngleEmbd / Codebook"
            if include_target:
                desc += " (Trips)"
            else:
                desc += " (Pairs)"
        super().__init__(name=name, desc=desc, **kwargs)
        self.n_embd = n_embd
        self.seq_shape = seq_shape
        self.include_target = include_target

    def build(self, input_shape):
        assert 'ctx/inp/vals' in input_shape, f"Input must have 'ctx/inp/vals' key, got {input_shape}"
        assert 'ctx/inp/idxs' in input_shape, f"Input must have 'ctx/inp/idxs' key, got {input_shape}"
        if self.include_target:
            assert 'ctx/tar/idxs' in input_shape, f"Input must have 'ctx/tar/idxs' key, got {input_shape}"

        self.angle_embd = AngleEmbd(n_embd=self.n_embd, name=f"{self.name}/angle_embd")

        self.inp_pos_embedders = [
            layers.Embedding(size, self.n_embd, name=f"{self.name}/inp_pos_embd_{i}")
            for i, size in enumerate(self.seq_shape)
        ]
        self.prepend_begin_pos = PrependToken(name=f"{self.name}/prepend_begin_pos")
        if self.include_target:
            self.tar_pos_embedders = [
                layers.Embedding(size, self.n_embd, name=f"{self.name}/tar_pos_embd_{i}")
                for i, size in enumerate(self.seq_shape)
            ]

    @tf.function
    @u.tf_scope
    def call(self, inputs):
        assert 'ctx/inp/vals' in inputs, f"Input must have 'ctx/inp/vals' key, got {u.tf_str(inputs)}"
        assert 'ctx/inp/idxs' in inputs, f"Input must have 'ctx/inp/idxs' key, got {u.tf_str(inputs)}"
        if self.include_target:
            assert 'ctx/tar/idxs' in inputs, f"Input must have 'ctx/tar/idxs' key, got {u.tf_str(inputs)}"

        vals = inputs['ctx/inp/vals']
        angle_embd = self.angle_embd(vals)

        inp_idxs = inputs['ctx/inp/idxs']
        inp_pos_embd = tf.add_n([
            embedder(inp_idxs[..., i])
            for i, embedder in enumerate(self.inp_pos_embedders)
        ])
        inp_pos_embd = self.prepend_begin_pos(inp_pos_embd)

        if self.include_target:
            tar_idxs = inputs['ctx/tar/idxs']
            tar_pos_embd = tf.add_n([
                embedder(tar_idxs[..., i])
                for i, embedder in enumerate(self.tar_pos_embedders)
            ])
            embd = (angle_embd + inp_pos_embd + tar_pos_embd) / tf.sqrt(tf.cast(3, u.dtype()))
        else:
            embd = (angle_embd + inp_pos_embd) / tf.sqrt(tf.cast(2, u.dtype()))

        return {
            'ctx/embd': embd,
        }


@export
class DecoderOnly_Value_Codebook(u.MxLayer):
    """
    Embedding for decoder-only model. Vals with AngleEmbd, Positions with Codebook.

    >>> embd = DecoderOnly_Angle_Codebook(n_embd=4, seq_shape=[5])
    >>> inps = {
    ...     'ctx/inp/vals': tf.constant([[[0.], [1.]]]), # angles
    ...     'ctx/inp/idxs': tf.constant([[[0], [1]]]),  # positions
    ... }
    >>> embd(inps)['ctx/embd'].shape
    TensorShape([1, 3, 4])
    >>> embd = DecoderOnly_Angle_Codebook(n_embd=4, seq_shape=[5], include_target=True)
    >>> inps = {
    ...     'ctx/inp/vals': tf.constant([[[0.], [1.]]]), # angles
    ...     'ctx/inp/idxs': tf.constant([[[0], [1]]]),  # positions
    ...     'ctx/tar/idxs': tf.constant([[[0], [1], [2]]]),  # positions
    ... }
    >>> embd(inps)['ctx/embd'].shape
    TensorShape([1, 3, 4])
    """

    def __init__(
        self,
        n_embd: int,
        seq_shape: list[int],
        include_target: bool = False,
        name="acotrips",
        desc=None,
        **kwargs
    ):
        if desc is None:
            desc = "AngleEmbd / Codebook"
            if include_target:
                desc += " (Trips)"
            else:
                desc += " (Pairs)"
        super().__init__(name=name, desc=desc, **kwargs)
        self.n_embd = n_embd
        self.seq_shape = seq_shape
        self.include_target = include_target

    def build(self, input_shape):
        assert 'ctx/inp/vals' in input_shape, f"Input must have 'ctx/inp/vals' key, got {input_shape}"
        assert 'ctx/inp/idxs' in input_shape, f"Input must have 'ctx/inp/idxs' key, got {input_shape}"
        if self.include_target:
            assert 'ctx/tar/idxs' in input_shape, f"Input must have 'ctx/tar/idxs' key, got {input_shape}"

        self.angle_embd = AngleEmbd(n_embd=self.n_embd, name=f"{self.name}/angle_embd")

        self.inp_pos_embedders = [
            layers.Embedding(size, self.n_embd, name=f"{self.name}/inp_pos_embd_{i}")
            for i, size in enumerate(self.seq_shape)
        ]
        self.prepend_begin_pos = PrependToken(name=f"{self.name}/prepend_begin_pos")
        if self.include_target:
            self.tar_pos_embedders = [
                layers.Embedding(size, self.n_embd, name=f"{self.name}/tar_pos_embd_{i}")
                for i, size in enumerate(self.seq_shape)
            ]

    @tf.function
    @u.tf_scope
    def call(self, inputs):
        assert 'ctx/inp/vals' in inputs, f"Input must have 'ctx/inp/vals' key, got {u.tf_str(inputs)}"
        assert 'ctx/inp/idxs' in inputs, f"Input must have 'ctx/inp/idxs' key, got {u.tf_str(inputs)}"
        if self.include_target:
            assert 'ctx/tar/idxs' in inputs, f"Input must have 'ctx/tar/idxs' key, got {u.tf_str(inputs)}"

        vals = inputs['ctx/inp/vals']
        angle_embd = self.angle_embd(vals)

        inp_idxs = inputs['ctx/inp/idxs']
        inp_pos_embd = tf.add_n([
            embedder(inp_idxs[..., i])
            for i, embedder in enumerate(self.inp_pos_embedders)
        ])
        inp_pos_embd = self.prepend_begin_pos(inp_pos_embd)

        if self.include_target:
            tar_idxs = inputs['ctx/tar/idxs']
            tar_pos_embd = tf.add_n([
                embedder(tar_idxs[..., i])
                for i, embedder in enumerate(self.tar_pos_embedders)
            ])
            embd = (angle_embd + inp_pos_embd + tar_pos_embd) / tf.sqrt(tf.cast(3, u.dtype()))
        else:
            embd = (angle_embd + inp_pos_embd) / tf.sqrt(tf.cast(2, u.dtype()))

        return {
            'ctx/embd': embd,
        }

# @export
# class AngleCodebookTriples(MxEmbedding):
#     """
#     Simple embedding for transformer regression over angles.

#     Embeds angles as unit vectors. Creates `n_repeats` copies of them, rotated
#     evenly around one quarter of the unit circle.

#     Embeds positions with a codebook.
#     """

#     def __init__(
#         self,
#         n_embd: int,
#         n_repeats: int,
#         name="congletrip",
#         desc="Angles: `n_repeats` rotations -> linear layer. Positions: Abs position codebook.",
#     ) -> None:
#         super().__init__(
#             n_embd=n_embd,
#             name=name,
#             desc=desc,
#         )
#         self.n_repeats = n_repeats
#         "Number of additional rotations of each angle to be added to the embedding."

#         self.task_config_type: Type[SeqEmbd_TaskConfig] = SeqEmbd_TaskConfig
#         self.task_cfg: SeqEmbd_TaskConfig | None = None

#     def configure(self, model: MxModel):
#         if isinstance(model, DecoderOnlyTransformer):
#             model.recieve_embd_config(model.embd_cfg_type(
#                 n_embd=self.n_embd,
#             ))
#         else:
#             raise NotImplementedError(f"Embedding {type_name(self)} does not support Model {type_name(model)}. If using autoreload in IPython, try restarting the interpreter.")

#     def make_embedder(self) -> Model:
#         "Creats the keras model for the embedding."

#         assert self.n_embd % 2 == 0, f"n_embd must be divisible by 2 to use angle embedding, got n_embd={self.n_embd}"
#         assert self.task_cfg is not None, "Must call task.configure(embedding) before embedding.make_embedder()."

#         initializer = lambda: tf.keras.initializers.TruncatedNormal(stddev=1/sqrt(self.n_embd))

#         inp_pos_embedder = layers.Embedding(self.task_cfg.sequence_length, self.n_embd, embeddings_initializer=initializer(), name="inp_pos_embedder")
#         tar_pos_embedder = layers.Embedding(self.task_cfg.sequence_length, self.n_embd, embeddings_initializer=initializer(), name="tar_pos_embedder")

#         # embed using index within current window
#         win_inp_pos_embedder = layers.Embedding(self.task_cfg.chunk_length, self.n_embd, embeddings_initializer=initializer(), name=f"win_inp_pos_embedder")
#         win_tar_pos_embedder = layers.Embedding(self.task_cfg.chunk_length, self.n_embd, embeddings_initializer=initializer(), name=f"win_tar_pos_embedder")

#         angle_embedder = make_angle_embedder(n_inp=self.task_cfg.n_input_dims, n_embd=self.n_embd, n_repeats=self.n_repeats)

#         prepend_begin_token = prepend_token(
#             token=tokens.BEGIN,
#             n_embd=self.n_embd,
#             name="prepend_begin_token",
#         )
#         def embed(inputs):

#             ## make angle embeddings
#             angles = inputs["context/vals"]

#             angle_passthrough, angle_embd = angle_embedder(angles)

#             window_positions = tf.tile(
#                 tf.range(shape(inputs["context/tar_idxs"])[1])[None],
#                 [shape(inputs["context/tar_idxs"])[0], 1],
#             )
#             win_inp_pos_embd = win_inp_pos_embedder(window_positions[:, :-1])
#             win_tar_pos_embd = win_tar_pos_embedder(window_positions)

#             ## make position embeddings
#             inp_pos_idxs = inputs["context/inp_idxs"][:, :, 0]
#             tp(inp_pos_idxs, "inp_pos_idxs")
#             inp_pos_embd = inp_pos_embedder(inp_pos_idxs)

#             tar_pos_idxs = inputs["context/tar_idxs"][:, :, 0]
#             tar_pos_embd = tar_pos_embedder(tar_pos_idxs)
#             tp(tar_pos_idxs, "tar_pos_idxs")

#             embd = win_tar_pos_embd + tar_pos_embd + prepend_begin_token(angle_embd + inp_pos_embd + win_inp_pos_embd)

#             # scale back to unit length
#             embd /= tf.sqrt(tf.cast(5, u.dtype()))


#             return angle_passthrough, embd


#         inputs = u.input_dict(
#             Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="context/vals"),
#             Input([None, 1],                          dtype=tf.int32,  name="context/inp_idxs"),
#             Input([None, 1],                          dtype=tf.int32,  name="context/tar_idxs"),
#         )
#         return Model(
#             inputs=inputs,
#             outputs=embed(inputs),
#             name=self.name,
#         ), inputs



@export
class AngleSinusoidal(MxEmbedding):
    """
    Simple embedding for transformer regression over angles.

    Embeds angles as unit vectors. Creates `n_repeats` copies of them, rotated
    evenly around one quarter of the unit circle.

    Embeds positions with a sinusoidal positional encoding.
    """

    def __init__(
        self,
        n_embd: int,
        n_repeats: int,
        name="angleembdsin",
        desc="Angles: `n_repeats` rotations -> linear layer. Positions: sinusoidal positional encoding.",
    ) -> None:
        super().__init__(
            n_embd=n_embd,
            name=name,
            desc=desc,
        )
        self.n_repeats = n_repeats
        "Number of additional rotations of each angle to be added to the embedding."

        self.task_config_type: Type[FloatEmbedding_TaskConfig] = FloatEmbedding_TaskConfig
        self.task_cfg: FloatEmbedding_TaskConfig | None = None

    def configure(self, model: MxModel):
        if isinstance(model, DecoderOnlyTransformer):
            model.recieve_embd_config(model.embd_cfg_type(
                n_embd=self.n_embd,
            ))
        else:
            raise NotImplementedError(f"Embedding {type_name(self)} does not support Model {type_name(model)}. If using autoreload in IPython, try restarting the interpreter.")

    def make_embedder(self) -> Model:
        "Creats the keras model for the embedding."

        assert self.n_embd % 2 == 0, f"n_embd must be divisible by 2 to use angle embedding, got n_embd={self.n_embd}"
        assert self.task_cfg is not None, "Must call task.configure(embedding) before embedding.make_embedder()."

        pos_embedder = positional_embedding(self.n_embd)
        dense_out = layers.Dense(self.n_embd, name="embd")

        prepend_begin_token = prepend_token(
            token=tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):

            ## make angle embeddings
            angles = inputs["context/vals"]

            angles = tf.cast(angles, u.dtype())

            angle_embd = ein.repeat(
                angles,
                "... seq feat -> ... seq (feat rep)",
                rep=self.n_repeats,
            )
            angle_embd /= tf.sqrt(tf.cast(shape(angle_embd)[-1], u.dtype()))

            ## make position embeddings
            pos_idxs = inputs["context/inp_idxs"]
            pos_embd = pos_embedder(pos_idxs)

            pos_embd /= tf.sqrt(tf.cast(shape(pos_embd)[-1]//2, u.dtype()))

            embd = prepend_begin_token(angle_embd + pos_embd)

            embd /= tf.sqrt(tf.cast(2, u.dtype()))

            return embd

        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="context/vals"),
            Input([None],                             dtype=tf.int32,   name="context/inp_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        ), inputs



@export
class AngleSinusoidalTriples(MxEmbedding):
    """
    Simple embedding for transformer regression over angles.

    Embeds angles as unit vectors. Creates `n_repeats` copies of them, rotated
    evenly around one quarter of the unit circle.

    Embeds positions with a sinudoidal positional encoding.
    """

    def __init__(
        self,
        n_embd: int,
        n_repeats: int,
        name="singletrip",
        desc="Angles: `n_repeats` rotations -> linear layer. Positions: sinudoidal positional encoding.",
    ) -> None:
        super().__init__(
            n_embd=n_embd,
            name=name,
            desc=desc,
        )
        self.n_repeats = n_repeats
        "Number of additional rotations of each angle to be added to the embedding."

        self.task_config_type: Type[SeqEmbd_TaskConfig] = SeqEmbd_TaskConfig
        self.task_cfg: SeqEmbd_TaskConfig | None = None

    def configure(self, model: MxModel):
        if isinstance(model, DecoderOnlyTransformer):
            model.recieve_embd_config(model.embd_cfg_type(
                n_embd=self.n_embd,
            ))
        else:
            raise NotImplementedError(f"Embedding {type_name(self)} does not support Model {type_name(model)}. If using autoreload in IPython, try restarting the interpreter.")

    def make_embedder(self) -> Model:
        "Creats the keras model for the embedding."

        assert self.n_embd % 2 == 0, f"n_embd must be divisible by 2 to use angle embedding, got n_embd={self.n_embd}"
        assert self.task_cfg is not None, "Must call task.configure(embedding) before embedding.make_embedder()."

        pos_embedder = positional_embedding(self.n_embd, name="pos_embd")

        angle_embedder = make_angle_embedder(n_inp=self.task_cfg.n_input_dims, n_embd=self.n_embd, n_repeats=self.n_repeats)

        prepend_begin_embd = prepend_token(
            token=tokens.BEGIN,
            n_embd=self.n_embd,
            name="prepend_begin_embd",
        )

        def embed(inputs):

            ## make angle embeddings
            angles = inputs["context/vals"]

            angle_passthrough, angle_embd = angle_embedder(angles)

            # # flatten to "embd" dim
            # angle_embd = dense_out(angles)

            ## make position embeddings
            inp_pos_idxs = inputs["context/inp_idxs"][:, :, 0]
            tp(inp_pos_idxs, "inp_pos_idxs")
            inp_pos_embd = pos_embedder(inp_pos_idxs)
            inp_pos_embd /= tf.sqrt(tf.cast(shape(inp_pos_embd)[-1], u.dtype())) # normalize

            tar_pos_idxs = inputs["context/tar_idxs"][:, :, 0]
            tar_pos_embd = pos_embedder(tar_pos_idxs)
            tar_pos_embd /= tf.sqrt(tf.cast(shape(tar_pos_embd)[-1], u.dtype())) # normalize to same scale as angle_embd
            tp(tar_pos_idxs, "tar_pos_idxs")

            embd = tar_pos_embd + prepend_begin_embd(angle_embd + inp_pos_embd)

            # scale back to unit length
            embd /= tf.sqrt(tf.cast(3, u.dtype()))

            return angle_passthrough, embd


        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="context/vals"),
            Input([None, 1],                          dtype=tf.int32,  name="context/inp_idxs"),
            Input([None, 1],                          dtype=tf.int32,  name="context/tar_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        ), inputs



@export
@dataclass
class CodebookMultidim_TaskConfig(FloatEmbedding_TaskConfig):
    seq_len: int
    """
    Max length of the flattened embedding sequence.
    """

    seq_dims: list[int]
    """
    Max value among seq_idxs for the respective dimensions.
    """

@export
class VectorCodebookMultidim(MxEmbedding):
    """
    Embedding for multi-dimensional sequences of vals.

    Takes
    -   1. a sequence of vectors with vals in the range [0, 1] and
    -   2. a sequence of multi-dimensional indices

    and produces a sequence of embeddings, where each embedding is a
    vector of length n_embd. The multi-dimensional indices are embedded
    with a codebook, and the scalars are embedded with a dense layer.
    """

    def __init__(
        self,
        n_embd: int,
        name="codebook",
        desc="Code-book based transformer embedding for multi-dimensional sequences.",
    ) -> None:
        super().__init__(
            n_embd=n_embd,
            name=name,
            desc=desc,
        )

        self.task_config_type: Type[CodebookMultidim_TaskConfig] = CodebookMultidim_TaskConfig
        self.task_cfg: CodebookMultidim_TaskConfig = None

    def configure(self, model: MxModel):

        if model.embd_cfg_type == Model_EmbeddingConfig:
            model.recieve_embd_config(model.embd_cfg_type(
                n_embd=self.n_embd,
            ))
        else:
            raise NotImplementedError(f"Embedding {type_name(self)} does not support Model {type_name(model)}. If using autoreload in IPython, try restarting the interpreter.")

    def make_embedder(self) -> Model:
        "Creats the keras model for the embedding."

        assert self.task_cfg is not None, "Must call task.configure(embedding) before embedding.make_embedder()."

        n_seq_dims = len(self.task_cfg.seq_dims)

        initializer = keras.initializers.TruncatedNormal(stddev=1./sqrt(self.n_embd))

        # embed using index within current window
        win_pos_embedder = layers.Embedding(self.task_cfg.seq_len, self.n_embd, name=f"win_pos_embd", embeddings_initializer=initializer)

        n_total = prod(self.task_cfg.seq_dims)

        abs_pos_embedder = layers.Embedding(n_total, self.n_embd, name=f"abs_pos_embd", embeddings_initializer=initializer)

        # dense_in = layers.Dense(self.n_embd, use_bias=False, name="val_embd")

        prepend_begin_token = prepend_token(
            token=tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):
            # print(inputs)
            ## make angle embeddings
            vals = inputs["context/vals"]

            val_embd = ein.repeat(vals, "... seq () -> ... seq embd", embd=self.n_embd)

            ## make abs position embeddings
            pos_idxs = inputs["context/inp_idxs"]
            # print(pos_idxs)
            # print(n_seq_dims)
            abs_pos_idxs = tf.add_n([
                pos_idxs[..., i] * prod(self.task_cfg.seq_dims[i+1:])
                for i in range(n_seq_dims)
            ])
            # print(abs_pos_idxs)
            pos_embd = abs_pos_embedder(abs_pos_idxs)

            ## make window position embeddings
            win_pos_idxs = tf.range(shape(pos_idxs)[1], dtype=tf.int32)
            win_pos_idxs = win_pos_idxs[None, :]
            win_pos_embd = win_pos_embedder(win_pos_idxs)

            embd = val_embd + win_pos_embd + pos_embd

            # scale back to unit length
            embd /= tf.sqrt(tf.cast(3, u.dtype()))

            return prepend_begin_token(embd)

        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="context/vals"),
            Input([None, n_seq_dims],                 dtype=tf.int32,  name="context/inp_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        ), inputs

@dataclass
class Multidim_TaskConfig(FloatEmbedding_TaskConfig):
    seq_dims: list[int]
    """
    Max value among seq_idxs for the respective dimensions.
    """

    chunk_length: int


@export
class VectorSinusoidalMultidim(MxEmbedding):
    """
    Embedding for multi-dimensional sequences of vals.

    Takes
    -   1. a sequence of vectors with vals in the range [0, 1] and
    -   2. a sequence of multi-dimensional indices

    and produces a sequence of embeddings, where each embedding is a
    vector of length n_embd. The multi-dimensional indices are embedded
    with a codebook, and the scalars are embedded with a dense layer.
    """

    def __init__(
        self,
        n_embd: int,
        name="codebook_sinusoidal",
        desc="Code-book based transformer embedding for multi-dimensional sequences.",
    ) -> None:
        super().__init__(
            n_embd=n_embd,
            name=name,
            desc=desc,
        )

        self.task_config_type: Type[Multidim_TaskConfig] = Multidim_TaskConfig
        self.task_cfg: Multidim_TaskConfig = None

    def configure(self, model: MxModel):
        if model.embd_cfg_type == Model_EmbeddingConfig:
            model.recieve_embd_config(model.embd_cfg_type(
                n_embd=self.n_embd,
            ))
        else:
            raise NotImplementedError(f"Embedding {type_name(self)} does not support Model {type_name(model)}. If using autoreload in IPython, try restarting the interpreter.")

    def make_embedder(self) -> Model:
        "Creats the keras model for the embedding."

        assert self.task_cfg is not None, "Must call task.configure(embedding) before embedding.make_embedder()."

        assert self.n_embd % (len(self.task_cfg.seq_dims) * 2) == 0, "n_embd must be divisible by 2D where D is the number of sequence dimensions."

        pos_embedders = [
            positional_embedding(self.n_embd // len(self.task_cfg.seq_dims), max_wavelength=10*seq_len, name=f"pos_embd_{i}")
            for i, seq_len in enumerate(self.task_cfg.seq_dims)
        ]
        dense_in = layers.Dense(self.n_embd, use_bias=True, name="val_embd")

        prepend_begin_token = prepend_token(
            token=tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):

            ## make angle embeddings
            vals = inputs["context/vals"]

            vals = tf.cast(vals, u.dtype())

            val_embd = dense_in(vals)

            ## make position embeddings
            pos_idxs = inputs["context/inp_idxs"]

            pos_embds = [
                pos_embedders[i](pos_idxs[:, :, i])
                for i in range(len(self.task_cfg.seq_dims))
            ]

            pos_embds = tf.concat(pos_embds, axis=-1)

            embd = val_embd + pos_embds

            return prepend_begin_token(embd)

        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims],  dtype=u.dtype(), name="context/vals"),
            Input([None, len(self.task_cfg.seq_dims)], dtype=tf.int32,  name="context/inp_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        ), inputs


@export
class DebugCodebookTriples(MxEmbedding):
    def __init__(
        self,
        n_embd: int,
        name="debugtriples",
        desc="Debug codebook (triples)",
    ) -> None:
        super().__init__(
            n_embd=n_embd,
            name=name,
            desc=desc,
        )

        self.task_config_type: Type[Multidim_TaskConfig] = Multidim_TaskConfig
        self.task_cfg: Multidim_TaskConfig = None

    def configure(self, model: MxModel):
        if model.embd_cfg_type == Model_EmbeddingConfig:
            model.recieve_embd_config(model.embd_cfg_type(
                n_embd=self.n_embd,
            ))
        else:
            raise NotImplementedError(f"Embedding {type_name(self)} does not support Model {type_name(model)}. If using autoreload in IPython, try restarting the interpreter.")

    def make_embedder(self) -> Model:
        "Creats the keras model for the embedding."

        assert self.task_cfg is not None, "Must call task.configure(embedding) before embedding.make_embedder()."

        assert self.n_embd % (len(self.task_cfg.seq_dims) * 2) == 0, "n_embd must be divisible by 2D where D is the number of sequence dimensions."

        inp_pos_embedder = layers.Embedding(prod(self.task_cfg.seq_dims), self.n_embd, name=f"{self.name}/inp_pos_embd")
        tar_pos_embedder = layers.Embedding(prod(self.task_cfg.seq_dims), self.n_embd, name=f"{self.name}/tar_pos_embd")
        dense_in = layers.Dense(self.n_embd, use_bias=True, name=f"{self.name}/val_embd")

        prepend_begin_token = prepend_token(
            token=tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):

            ## make angle embeddings
            vals = inputs["context/vals"]

            vals = tf.cast(vals, u.dtype())

            val_embd = dense_in(vals)

            ## make position embeddings
            inp_idxs = inputs["context/inp_idxs"]
            inp_idxs = u.multidim_idxs_to_flat_idxs(inp_idxs, self.task_cfg.seq_dims)
            inp_pos_embd = inp_pos_embedder(inp_idxs)

            inp_embd = prepend_begin_token(val_embd + inp_pos_embd)

            tar_idxs = inputs["context/tar_idxs"]
            tar_idxs =  u.multidim_idxs_to_flat_idxs(tar_idxs, self.task_cfg.seq_dims)
            tar_pos_embd = tar_pos_embedder(tar_idxs)

            embd = tar_pos_embd + inp_embd

            return embd

        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims],  dtype=u.dtype(), name="context/vals"),
            Input([None, len(self.task_cfg.seq_dims)], dtype=tf.int32,  name="context/inp_idxs"),
            Input([None, len(self.task_cfg.seq_dims)], dtype=tf.int32,  name="context/tar_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        ), inputs


@dataclass
class DiscreteMultidimEmbd_TaskConfig(DiscreteEmbedding_TaskConfig):
    chunk_length: int
    seq_dims: list[int]

@export
class DebugCodebookCodebookTriples(MxEmbedding):
    def __init__(
        self,
        n_embd: int,
        name="dubcodetriples",
        desc="Codebook/Codebook (triples)",
    ) -> None:
        super().__init__(
            n_embd=n_embd,
            name=name,
            desc=desc,
        )

        self.task_config_type: Type[DiscreteMultidimEmbd_TaskConfig] = DiscreteMultidimEmbd_TaskConfig
        self.task_cfg: DiscreteMultidimEmbd_TaskConfig = None

    def configure(self, model: MxModel):
        if model.embd_cfg_type == Model_EmbeddingConfig:
            model.recieve_embd_config(model.embd_cfg_type(
                n_embd=self.n_embd,
            ))
        else:
            raise NotImplementedError(f"Embedding {type_name(self)} does not support Model {type_name(model)}. If using autoreload in IPython, try restarting the interpreter.")

    def make_embedder(self) -> Model:
        "Creats the keras model for the embedding."

        assert self.task_cfg is not None, "Must call task.configure(embedding) before embedding.make_embedder()."

        val_embedder = layers.Embedding(self.task_cfg.n_tokens, self.n_embd, name=f"{self.name}/val_embd")
        inp_pos_embedder = layers.Embedding(prod(self.task_cfg.seq_dims), self.n_embd, name=f"{self.name}/inp_pos_embd")
        tar_pos_embedder = layers.Embedding(prod(self.task_cfg.seq_dims), self.n_embd, name=f"{self.name}/tar_pos_embd")

        prepend_begin_token = prepend_token(
            token=tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):

            ## make angle embeddings
            tokens = inputs["context/tokens"]

            val_embd = val_embedder(tokens)

            ## make position embeddings
            inp_idxs = inputs["context/inp_idxs"]
            inp_idxs = u.multidim_idxs_to_flat_idxs(inp_idxs, self.task_cfg.seq_dims)
            inp_pos_embd = inp_pos_embedder(inp_idxs)

            inp_embd = prepend_begin_token(val_embd + inp_pos_embd)

            tar_idxs = inputs["context/tar_idxs"]
            tar_idxs =  u.multidim_idxs_to_flat_idxs(tar_idxs, self.task_cfg.seq_dims)
            tar_pos_embd = tar_pos_embedder(tar_idxs)

            embd = tar_pos_embd + inp_embd

            return embd

        inputs = u.input_dict(
            Input([None],                              dtype=tf.int32, name="context/tokens"),
            Input([None, len(self.task_cfg.seq_dims)], dtype=tf.int32, name="context/inp_idxs"),
            Input([None, len(self.task_cfg.seq_dims)], dtype=tf.int32, name="context/tar_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        ), inputs


@export
class DebugCodebook(MxEmbedding):
    def __init__(
        self,
        n_embd: int,
        name="debugembd",
        desc="Debug codebook.",
    ) -> None:
        super().__init__(
            n_embd=n_embd,
            name=name,
            desc=desc,
        )

        self.task_config_type: Type[Multidim_TaskConfig] = Multidim_TaskConfig
        self.task_cfg: Multidim_TaskConfig = None

    def configure(self, model: MxModel):
        if model.embd_cfg_type == Model_EmbeddingConfig:
            model.recieve_embd_config(model.embd_cfg_type(
                n_embd=self.n_embd,
            ))
        else:
            raise NotImplementedError(f"Embedding {type_name(self)} does not support Model {type_name(model)}. If using autoreload in IPython, try restarting the interpreter.")

    def make_embedder(self) -> Model:
        "Creats the keras model for the embedding."

        assert self.task_cfg is not None, "Must call task.configure(embedding) before embedding.make_embedder()."

        assert self.n_embd % (len(self.task_cfg.seq_dims) * 2) == 0, "n_embd must be divisible by 2D where D is the number of sequence dimensions."

        pos_embedder = layers.Embedding(prod(self.task_cfg.seq_dims), self.n_embd, name="pos_embd")
        dense_in = layers.Dense(self.n_embd, use_bias=True, name="val_embd")

        prepend_begin_token = prepend_token(
            token=tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):

            ## make angle embeddings
            vals = inputs["context/vals"]

            vals = tf.cast(vals, u.dtype())

            val_embd = dense_in(vals)

            ## make position embeddings
            pos_idxs = inputs["context/inp_idxs"]
            pos_idxs = u.multidim_idxs_to_flat_idxs(pos_idxs, self.task_cfg.seq_dims)
            pos_embd = pos_embedder(pos_idxs)

            embd = val_embd + pos_embd

            return prepend_begin_token(embd)

        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims],  dtype=u.dtype(), name="context/vals"),
            Input([None, len(self.task_cfg.seq_dims)], dtype=tf.int32,  name="context/inp_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        ), inputs


# if __name__ == "__main__":
#     u.set_debug(True)

#     ## test TransformerAngleVectorEmbedding
#     data = Dataset.from_tensor_slices({
#         "angles": tf.random.uniform((33, 10, 6), dtype=tf.float32),
#         "seq_idxs": tf.random.uniform((33, 10), maxval=10, dtype=tf.int32),
#     })
#     data = DSets(
#         train=data.take(9),
#         val=data.skip(9).take(9),
#         test=data.skip(18),
#     )
#     data = data.batch(7, 13)

#     embedding = AngleCodebook(
#         n_embd=32,
#         n_repeats=4,
#     )
#     model = DecoderOnlyTransformer(
#         n_layers=2,
#         n_heads=2,
#         n_hidden=32,
#     )

#     embedding.configure(model)
#     embedding.receive_task_config(embedding.task_config_type(
#         n_input_dims=6,
#         sequence_length=10,
#         chunk_length=10,
#     ))

#     dbg(data, "data")
#     embedder, inputs = embedding.make_embedder()
#     for x in data.train:
#         dbg(embedder(dbg(x, "embd input")), "embd output")


#     ## test TransformerMultidim
#     data = Dataset.from_tensor_slices({
#         "vals": tf.random.uniform((33, 10, 10, 6), dtype=tf.float32),
#         "seq_idxs": tf.random.uniform((33, 10, 10, 2), maxval=10, dtype=tf.int32),
#     })
#     data = DSets(
#         train=data.take(9),
#         val=data.skip(9).take(9),
#         test=data.skip(18),
#     )
#     data = data.batch(7, 13)

#     data = dbg(dbg(data, "data in").map(lambda x: {
#         "vals": ein.rearrange(x["vals"], "b s1 s2 f -> b (s1 s2) f"),
#         "seq_idxs": ein.rearrange(x["seq_idxs"], "b s1 s2 i -> b (s1 s2) i"),
#     }), "data out")

#     embedding = VectorSinusoidalMultidim(
#         n_embd=32,
#     )
#     model = DecoderOnlyTransformer(
#         n_layers=2,
#         n_heads=2,
#         n_hidden=32,
#     )

#     embedding.configure(model)

#     embedding.receive_task_config(embedding.task_config_type(
#         n_input_dims=6,
#         seq_dims=[10, 10],
#     ))

#     dbg(data, "data")
#     embedder, inputs = embedding.make_embedder()
#     for x in data.train:
#         dbg(embedder(dbg(x, "embd input")), "embd output")


if __name__ == '__main__':
    import doctest
    doctest.testmod()
