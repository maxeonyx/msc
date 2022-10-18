from __future__ import annotations
from mx.models.decoder_only import Resnet

from mx.prelude import *
from mx.models import DecoderOnlyTransformer
from mx.pipeline import MxEmbedding, Embedding_TaskConfig, Model_EmbeddingConfig, MxModel
import mx.layers as mxl
from mx.utils import DSets, dtype

@export
@dataclass
class AngleVectorSequence_TaskConfig(Embedding_TaskConfig):
    sequence_length: int
    """
    Max length of the sequence to be embedded.
    Max value among seq_idxs.
    """

@export
class AngleVectorSequence(MxEmbedding):
    """
    Simple embedding for transformer regression over angles.

    Embeds angles as unit vectors. Creates `n_repeats` copies of them, rotated
    evenly around one quarter of the unit circle.

    Embeds positions with a codebook.
    """

    def __init__(self, n_embd: int, n_repeats: int) -> None:
        super().__init__(
            desc="TransformerAngleVectorEmbedding",
            name="transformer_angle_vector",
            n_embd=n_embd
        )
        self.n_repeats = n_repeats
        "Number of additional rotations of each angle to be added to the embedding."

        self.task_config_type: Type[AngleVectorSequence_TaskConfig] = AngleVectorSequence_TaskConfig
        self.task_cfg: AngleVectorSequence_TaskConfig | None = None

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
        prepend_begin_token = mxl.prepend_token(
            token=mxl.tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):

            ## make angle embeddings
            angles = inputs["angles"]

            angles = tf.cast(angles, u.dtype())

            scale = (tau / 4.) * (1. / self.n_repeats) # only need to produce rotations up to tau/4, because the model can easily invert angles
            offsets = tf.cast(tf.range(self.n_repeats), dtype=u.dtype()) * scale
            # add "repeats" dim
            angles = angles[..., None]
            angles = angles + tf.broadcast_to(offsets, tf.broadcast_dynamic_shape(tf.shape(offsets), tf.shape(angles)))
            # add "sincos" dim
            angles = tf.stack([tf.sin(angles), tf.cos(angles)], axis=-1)
            angles = ein.rearrange(angles, "... seq feat rep sincos -> ... seq (feat rep sincos)")
            # flatten to "embd" dim
            angle_embd = dense_out(angles)

            ## make position embeddings
            pos_idxs = inputs["seq_idxs"]
            pos_embd = pos_embedder(pos_idxs)

            return prepend_begin_token(angle_embd + pos_embd)

        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="angles"),
            Input([None],                             dtype=tf.int32,   name="seq_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name="TransformerAngleVectorEmbedding",
        )


@export
class AngleSinusoidalSequence(MxEmbedding):
    """
    Simple embedding for transformer regression over angles.

    Embeds angles as unit vectors. Creates `n_repeats` copies of them, rotated
    evenly around one quarter of the unit circle.

    Embeds positions with a sinusoidal positional encoding.
    """

    def __init__(self, n_embd: int, n_repeats: int) -> None:
        super().__init__(
            desc="TransformerAngleVectorEmbedding",
            name="transformer_angle_vector",
            n_embd=n_embd
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

        pos_embedder = mxl.positional_embedding(self.n_embd)
        dense_out = layers.Dense(self.n_embd, name="embd")

        import mx.layers as mxl
        prepend_begin_token = mxl.prepend_token(
            token=mxl.tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):

            ## make angle embeddings
            angles = inputs["angles"]

            angles = tf.cast(angles, u.dtype())

            scale = (tau / 4.) * (1. / self.n_repeats) # only need to produce rotations up to tau/4, because the model can easily invert angles
            offsets = tf.cast(tf.range(self.n_repeats), dtype=u.dtype()) * scale
            # add "repeats" dim
            angles = angles[..., None]
            angles = angles + tf.broadcast_to(offsets, tf.broadcast_dynamic_shape(tf.shape(offsets), tf.shape(angles)))
            # add "sincos" dim
            angles = tf.stack([tf.sin(angles), tf.cos(angles)], axis=-1)
            angles = ein.rearrange(angles, "... seq feat rep sincos -> ... seq (feat rep sincos)")
            # flatten to "embd" dim
            angle_embd = dense_out(angles)

            ## make position embeddings
            pos_idxs = inputs["seq_idxs"]
            pos_embd = pos_embedder(pos_idxs)

            return prepend_begin_token(angle_embd + pos_embd)

        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims], dtype=u.dtype(), name="angles"),
            Input([None],                             dtype=tf.int32,   name="seq_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name="TransformerAngleVectorEmbedding",
        )

@export
@dataclass
class VectorCodebookMultidim_TaskConfig(Embedding_TaskConfig):
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

    def __init__(self, n_embd: int) -> None:
        super().__init__(
            desc="Code-book based transformer embedding for multi-dimensional sequences.",
            name="codebook",
            n_embd=n_embd
        )

        self.task_config_type: Type[VectorCodebookMultidim_TaskConfig] = VectorCodebookMultidim_TaskConfig
        self.task_cfg: VectorCodebookMultidim_TaskConfig = None

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

        prepend_begin_token = mxl.prepend_token(
            token=mxl.tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):
            print(inputs)
            ## make angle embeddings
            vals = inputs["values"]

            val_embd = ein.repeat(vals, "... seq () -> ... seq embd", embd=self.n_embd)

            ## make abs position embeddings
            pos_idxs = inputs["seq_idxs"]
            print(pos_idxs)
            print(n_seq_dims)
            abs_pos_idxs = tf.add_n([
                pos_idxs[..., i] * prod(self.task_cfg.seq_dims[i+1:])
                for i in range(n_seq_dims)
            ])
            print(abs_pos_idxs)
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
            Input([None, self.task_cfg.n_input_dims],  dtype=u.dtype(), name="values"),
            Input([None, n_seq_dims], dtype=tf.int32,   name="seq_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        )

@dataclass
class Multidim_TaskConfig(Embedding_TaskConfig):
    seq_dims: list[int]
    """
    Max value among seq_idxs for the respective dimensions.
    """


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

    def __init__(self, n_embd: int) -> None:
        super().__init__(
            desc="Code-book based transformer embedding for multi-dimensional sequences.",
            name="codebook_sinusoidal",
            n_embd=n_embd
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
            mxl.positional_embedding(self.n_embd // len(self.task_cfg.seq_dims), max_wavelength=10*seq_len, name=f"pos_embd_{i}")
            for i, seq_len in enumerate(self.task_cfg.seq_dims)
        ]
        dense_in = layers.Dense(self.n_embd, use_bias=True, name="val_embd")

        prepend_begin_token = mxl.prepend_token(
            token=mxl.tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):

            ## make angle embeddings
            vals = inputs["values"]

            vals = tf.cast(vals, u.dtype())

            val_embd = dense_in(vals)

            ## make position embeddings
            pos_idxs = inputs["seq_idxs"]

            pos_embds = [
                pos_embedders[i](pos_idxs[:, :, i])
                for i in range(len(self.task_cfg.seq_dims))
            ]

            pos_embds = tf.concat(pos_embds, axis=-1)

            embd = val_embd + pos_embds

            return prepend_begin_token(embd)

        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims],  dtype=u.dtype(), name="values"),
            Input([None, len(self.task_cfg.seq_dims)], dtype=tf.int32,  name="seq_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        )


@export
class DebugCodebook(MxEmbedding):
    def __init__(self, n_embd: int) -> None:
        super().__init__(
            desc="Debug codebook.",
            name="debugembd",
            n_embd=n_embd
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

        prepend_begin_token = mxl.prepend_token(
            token=mxl.tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):

            ## make angle embeddings
            vals = inputs["values"]

            vals = tf.cast(vals, u.dtype())

            val_embd = dense_in(vals)

            ## make position embeddings
            pos_idxs = inputs["seq_idxs"]
            pos_idxs = u.multidim_idxs_to_flat_idxs(pos_idxs, self.task_cfg.seq_dims)
            pos_embd = pos_embedder(pos_idxs)

            embd = val_embd + pos_embd

            return prepend_begin_token(embd)

        inputs = u.input_dict(
            Input([None, self.task_cfg.n_input_dims],  dtype=u.dtype(), name="values"),
            Input([None, len(self.task_cfg.seq_dims)], dtype=tf.int32,  name="seq_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name=self.name,
        )


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

    embedding = AngleVectorSequence(
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
    ))

    dbg(data)
    embedder = embedding.make_embedder()
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

    dbg(data)
    embedder = embedding.make_embedder()
    for x in data.train:
        dbg(embedder(dbg(x, "embd input")), "embd output")
