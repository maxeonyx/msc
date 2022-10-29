from __future__ import annotations
from enum import Enum

from mx.prelude import *
from mx.models import DecoderOnlyTransformer
from mx.pipeline import MxEmbedding, Embedding_TaskConfig, Model_EmbeddingConfig, MxModel
import mx.layers as mxl
from mx.utils import DSets, Einshape

@export
def codebook(n_tokens: int, embd_shape: Einshape, add_begin_token: bool = True, name="codebook") -> Model:

    embedder = layers.Embedding(n_tokens, embd_shape.f_product, name=f"{name}/embd")

    def call(tokens):

        embd = embedder(tokens)

        return embd

    inputs = u.input_dict(
        Input(shape=embd_shape.s_shape, dtype=tf.int32, name="tokens"),
    )

    return Model(inputs=inputs, outputs=call(**inputs), name=name)

@export
class tokens(Enum):
    BEGIN_VAL = 0
    BEGIN = 1
    END   = 2

@export
def prepend_token(token: tokens, n_embd: int, name="prepend_token") -> Model:

    assert token in tokens, f"Unknown token {tokens!r}, must be one of {list(tokens.__members__)!r}"

    initializer = lambda: tf.keras.initializers.TruncatedNormal(stddev=1/sqrt(n_embd))

    token_embedding = layers.Embedding(1, n_embd, embeddings_initializer=initializer(), name=f"{name}/begin_embd")

    tf_token = tf.constant(token.value, tf.int32)

    def call(input):
        embd = inputs["embd"]
        batch_size = shape(embd)[0]

        tokens = tf.repeat(tf_token[None], batch_size[None])
        tokens = tokens[:, None]
        token_embd = token_embedding(tokens)
        embd = tf.concat([token_embd, embd], axis=1) # concat along first sequence dim
        return embd

    inputs = u.input_dict(
        Input(shape=[None, n_embd], dtype=u.dtype(), name="embd"),
    )

    return Model(inputs=inputs, outputs=call(inputs), name=name)





@export
def positional_embedding(n_embd, max_wavelength=10000, name="embd") -> Model:
    assert n_embd % 2 == 0, f"embd_dim must be divisible by 2 to use positional encoding, got embd_dim={n_embd}"

    # based from the keras source code
    # https://github.com/keras-team/keras-nlp/blob/v0.3.0/keras_nlp/layers/sine_position_encoding.py#L21

    @u.tf_scope
    def positional_encoding(inputs):
        idxs = inputs["idxs"]
        position = tf.cast(idxs, u.dtype())
        min_freq = 1. / max_wavelength
        timescales = tf.pow(
            min_freq,
            tf.range(n_embd, dtype=tf.float32) / n_embd
        )
        timescales = tf.cast(timescales, u.dtype())
        position = ein.rearrange(position, '... seq -> ... seq ()')
        timescales = ein.rearrange(timescales, '... embd -> ... () embd')
        angles = position * timescales
        # even indices are sine, odd are cosine
        cos_mask = tf.cast(tf.range(n_embd) % 2, u.dtype())
        sin_mask = 1 - cos_mask
        # embedding shape is [seq_length, hidden_size]
        positional_encodings = (
            tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
        )

        # scale norm. because we use sin/cos we scale by 1/sqrt(D/2) instead of 1/sqrt(D)
        positional_encodings *= tf.cast(tf.math.sqrt(tf.cast(n_embd // 2, u.dtype())), u.dtype())

        return positional_encodings

    inputs = u.input_dict(
        Input(shape=[None], dtype=tf.int32, name="idxs"),
    )

    return Model(inputs=inputs, outputs=positional_encoding(inputs), name=name)



@export
@dataclass
class SeqEmbd_TaskConfig(Embedding_TaskConfig):
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
            angles = inputs["context/values"]

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
            Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="context/values"),
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
            angles = inputs["context/values"]

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
            Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="context/values"),
            Input([None, 1],                          dtype=tf.int32,  name="context/inp_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        ), inputs

def make_angle_embedder(n_inp, n_embd, n_repeats, name="angle_embd"):

    prepend_begin_val = prepend_token(token=tokens.BEGIN, n_embd=n_inp*2, name=f"{name}/prepend_begin_val")

    scale = (tau / 4.) * (1. / n_repeats) # only need to produce rotations up to tau/4, because the model can easily invert angles
    offsets = tf.cast(tf.range(n_repeats), dtype=u.dtype()) * scale

    dense_in = layers.Dense(n_embd, name=f"{name}/dense")

    def embed_angles(angles):

        angles = tf.cast(angles, u.dtype())
        angles = tf.stack([tf.sin(angles), tf.cos(angles)], axis=-1)
        angles = ein.rearrange(
            angles,
            "... feat sincos -> ... (feat sincos)",
        )

        angle_passthrough = prepend_begin_val(angles)

        angles = ein.repeat(
            angles,
            "... -> ... rep",
            rep=n_repeats,
        ) + offsets

        angles = ein.rearrange(
            angles,
            "... feat rep -> ... (feat rep)",
        )

        # flatten to "embd" dim
        angle_embd = dense_in(angles)

        angle_embd /= tf.sqrt(tf.cast(angles.shape[-1]//2, u.dtype()))

        return angle_passthrough, angle_embd

    return embed_angles


@export
class AngleCodebookTriples(MxEmbedding):
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
        name="congletrip",
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

        initializer = lambda: tf.keras.initializers.TruncatedNormal(stddev=1/sqrt(self.n_embd))

        inp_pos_embedder = layers.Embedding(self.task_cfg.sequence_length, self.n_embd, embeddings_initializer=initializer(), name="inp_pos_embedder")
        tar_pos_embedder = layers.Embedding(self.task_cfg.sequence_length, self.n_embd, embeddings_initializer=initializer(), name="tar_pos_embedder")

        # embed using index within current window
        win_inp_pos_embedder = layers.Embedding(self.task_cfg.chunk_length, self.n_embd, embeddings_initializer=initializer(), name=f"win_inp_pos_embedder")
        win_tar_pos_embedder = layers.Embedding(self.task_cfg.chunk_length, self.n_embd, embeddings_initializer=initializer(), name=f"win_tar_pos_embedder")

        angle_embedder = make_angle_embedder(n_inp=self.task_cfg.n_input_dims, n_embd=self.n_embd, n_repeats=self.n_repeats)

        prepend_begin_token = prepend_token(
            token=tokens.BEGIN,
            n_embd=self.n_embd,
            name="prepend_begin_token",
        )
        def embed(inputs):

            ## make angle embeddings
            angles = inputs["context/values"]

            angle_passthrough, angle_embd = angle_embedder(angles)

            window_positions = tf.tile(
                tf.range(shape(inputs["context/tar_idxs"])[1])[None],
                [shape(inputs["context/tar_idxs"])[0], 1],
            )
            win_inp_pos_embd = win_inp_pos_embedder(window_positions[:, :-1])
            win_tar_pos_embd = win_tar_pos_embedder(window_positions)

            ## make position embeddings
            inp_pos_idxs = inputs["context/inp_idxs"][:, :, 0]
            tp(inp_pos_idxs, "inp_pos_idxs")
            inp_pos_embd = inp_pos_embedder(inp_pos_idxs)

            tar_pos_idxs = inputs["context/tar_idxs"][:, :, 0]
            tar_pos_embd = tar_pos_embedder(tar_pos_idxs)
            tp(tar_pos_idxs, "tar_pos_idxs")

            embd = win_tar_pos_embd + tar_pos_embd + prepend_begin_token(angle_embd + inp_pos_embd + win_inp_pos_embd)

            # scale back to unit length
            embd /= tf.sqrt(tf.cast(5, u.dtype()))


            return angle_passthrough, embd


        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="context/values"),
            Input([None, 1],                          dtype=tf.int32,  name="context/inp_idxs"),
            Input([None, 1],                          dtype=tf.int32,  name="context/tar_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        ), inputs


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

        self.task_config_type: Type[Embedding_TaskConfig] = Embedding_TaskConfig
        self.task_cfg: Embedding_TaskConfig | None = None

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
            angles = inputs["context/values"]

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
            Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="context/values"),
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
            angles = inputs["context/values"]

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
            Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="context/values"),
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
class CodebookMultidim_TaskConfig(Embedding_TaskConfig):
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
    Embedding for multi-dimensional sequences of values.

    Takes
    -   1. a sequence of vectors with values in the range [0, 1] and
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
            vals = inputs["context/values"]

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
            Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="context/values"),
            Input([None, n_seq_dims],                 dtype=tf.int32,  name="context/inp_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        ), inputs

@dataclass
class Multidim_TaskConfig(Embedding_TaskConfig):
    seq_dims: list[int]
    """
    Max value among seq_idxs for the respective dimensions.
    """

    chunk_length: int


@export
class VectorSinusoidalMultidim(MxEmbedding):
    """
    Embedding for multi-dimensional sequences of values.

    Takes
    -   1. a sequence of vectors with values in the range [0, 1] and
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
            vals = inputs["context/values"]

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
            Input([None, self.task_cfg.n_input_dims],  dtype=u.dtype(), name="context/values"),
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
            vals = inputs["context/values"]

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
            Input([None, self.task_cfg.n_input_dims],  dtype=u.dtype(), name="context/values"),
            Input([None, len(self.task_cfg.seq_dims)], dtype=tf.int32,  name="context/inp_idxs"),
            Input([None, len(self.task_cfg.seq_dims)], dtype=tf.int32,  name="context/tar_idxs"),
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
            vals = inputs["context/values"]

            vals = tf.cast(vals, u.dtype())

            val_embd = dense_in(vals)

            ## make position embeddings
            pos_idxs = inputs["context/inp_idxs"]
            pos_idxs = u.multidim_idxs_to_flat_idxs(pos_idxs, self.task_cfg.seq_dims)
            pos_embd = pos_embedder(pos_idxs)

            embd = val_embd + pos_embd

            return prepend_begin_token(embd)

        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims],  dtype=u.dtype(), name="context/values"),
            Input([None, len(self.task_cfg.seq_dims)], dtype=tf.int32,  name="context/inp_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        ), inputs


if __name__ == "__main__":
    u.set_debug(True)

    ## test TransformerAngleVectorEmbedding
    data = Dataset.from_tensor_slices({
        "angles": tf.random.uniform((33, 10, 6), dtype=tf.float32),
        "seq_idxs": tf.random.uniform((33, 10), maxval=10, dtype=tf.int32),
    })
    data = DSets(
        train=data.take(9),
        val=data.skip(9).take(9),
        test=data.skip(18),
    )
    data = data.batch(7, 13)

    embedding = AngleCodebook(
        n_embd=32,
        n_repeats=4,
    )
    model = DecoderOnlyTransformer(
        n_layers=2,
        n_heads=2,
        n_hidden=32,
    )

    embedding.configure(model)
    embedding.receive_task_config(embedding.task_config_type(
        n_input_dims=6,
        sequence_length=10,
        chunk_length=10,
    ))

    dbg(data, "data")
    embedder, inputs = embedding.make_embedder()
    for x in data.train:
        dbg(embedder(dbg(x, "embd input")), "embd output")


    ## test TransformerMultidim
    data = Dataset.from_tensor_slices({
        "values": tf.random.uniform((33, 10, 10, 6), dtype=tf.float32),
        "seq_idxs": tf.random.uniform((33, 10, 10, 2), maxval=10, dtype=tf.int32),
    })
    data = DSets(
        train=data.take(9),
        val=data.skip(9).take(9),
        test=data.skip(18),
    )
    data = data.batch(7, 13)

    data = dbg(dbg(data, "data in").map(lambda x: {
        "values": ein.rearrange(x["values"], "b s1 s2 f -> b (s1 s2) f"),
        "seq_idxs": ein.rearrange(x["seq_idxs"], "b s1 s2 i -> b (s1 s2) i"),
    }), "data out")

    embedding = VectorSinusoidalMultidim(
        n_embd=32,
    )
    model = DecoderOnlyTransformer(
        n_layers=2,
        n_heads=2,
        n_hidden=32,
    )

    embedding.configure(model)

    embedding.receive_task_config(embedding.task_config_type(
        n_input_dims=6,
        seq_dims=[10, 10],
    ))

    dbg(data, "data")
    embedder, inputs = embedding.make_embedder()
    for x in data.train:
        dbg(embedder(dbg(x, "embd input")), "embd output")
