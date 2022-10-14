from __future__ import annotations
from mx.prelude import *

from mx.models import DecoderOnlyTransformer
import mx.layers as mxl
from mx.pipeline import Embedding, MxModel

@export
class TransformerMultidimScalar(Embedding):
    """
    Embedding for multi-dimensional sequences of scalars.

    Takes
    -   1. a sequence of scalars in the range [0, 1] and
    -   2. a sequence of multi-dimensional indices

    and produces a sequence of embeddings, where each embedding is a
    vector of length n_embd. The multi-dimensional indices are embedded
    with a codebook, and the scalars are embedded with a dense layer.
    """

    @dataclass
    class TaskSpecificConfig(Embedding.TaskSpecificConfig):
        sequence_lengths: list[int]
        """
        Max length of the sequences to be embedded.
        Max value among seq_idxs for the respective dimensions.
        """

    def __init__(self, n_embd: int, n_repeats: int, n_seq_dims: int) -> None:
        super().__init__(
            name="TransformerAngleVectorEmbedding",
            identifier="transformer_angle_vector",
            n_embd=n_embd
        )

        assert n_embd % n_seq_dims == 0, f"n_embd must be divisible by n_seq_dims, got n_embd={n_embd}, n_seq_dims={n_seq_dims}"

        self.n_repeats = n_repeats
        "Number of additional rotations of each angle to be added to the embedding."

        self.n_seq_dims = n_seq_dims
        "Number of dimensions of the sequence to be embedded."

        self.task_config_type: Type[TransformerMultidimScalar.TaskSpecificConfig] = TransformerMultidimScalar.TaskSpecificConfig
        self.task_cfg: TransformerMultidimScalar.TaskSpecificConfig | None = None

    def configure(self, model: MxModel):
        if isinstance(model, DecoderOnlyTransformer):
            model.recieve_embd_config(model.embd_cfg_type(
                n_embd=self.n_embd,
            ))
        else:
            raise NotImplementedError(f"{type_name(self).__name__} does not support {type(model)}")

    def make_embedder(self) -> Model:
        "Creats the keras model for the embedding."

        assert self.task_cfg is not None, "Must call task.configure(embedding) before embedding.make_embedder()."

        pos_embedders = [
            tf.keras.layers.Embedding(seq_len, self.n_embd, name=f"idx_embd_{i}")
            for i, seq_len in enumerate(self.task_cfg.sequence_lengths)
        ]
        dense_in = tf.keras.layers.Dense(self.n_embd, use_bias=True, name="val_embd")

        prepend_begin_token = mxl.prepend_token(
            token=mxl.tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):

            ## make angle embeddings
            vals = inputs["values"]
            val_embd = dense_in(vals)

            ## make position embeddings
            pos_idxs = inputs["seq_idxs"]

            pos_embds = [
                pos_embedders[i](pos_idxs[:, :, i])
                for i in range(self.n_seq_dims)
            ]

            embd = tf.math.add_n([val_embd, *pos_embds])

            return prepend_begin_token(embd)

        inputs = u.input_dict(
            Input([None],                  dtype=tf.float32, name="values"),
            Input([None, self.n_seq_dims], dtype=tf.int32,   name="seq_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name="TransformerMultidimScalar",
        )
