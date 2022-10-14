from mx.prelude import *

from mx.models import DecoderOnlyTransformer

from mx.pipeline import Embedding, MxModel

@export
class TransformerAngleVectorEmbedding(Embedding):
    """
    Simple embedding for transformer regression over angles.

    Embeds angles as unit vectors. Creates `n_repeats` copies of them, rotated
    evenly around one quarter of the unit circle.

    Embeds positions with a codebook.
    """

    @dataclass
    class TaskSpecificConfig(Embedding.TaskSpecificConfig):
        sequence_length: int
        """
        Max length of the sequence to be embedded.
        Max value among seq_idxs.
        """

    def __init__(self, n_embd: int, n_repeats: int) -> None:
        super().__init__(
            name="TransformerAngleVectorEmbedding",
            identifier="transformer_angle_vector",
            n_embd=n_embd
        )
        self.n_repeats = n_repeats
        "Number of additional rotations of each angle to be added to the embedding."

        self.task_config_type: Type[TransformerAngleVectorEmbedding.TaskSpecificConfig] = TransformerAngleVectorEmbedding.TaskSpecificConfig
        self.task_cfg: TransformerAngleVectorEmbedding.TaskSpecificConfig | None = None

    def configure(self, model: MxModel):
        if isinstance(model, DecoderOnlyTransformer):
            model.recieve_embd_config(model.embd_cfg_type(
                n_embd=self.n_embd,
            ))
        else:
            raise NotImplementedError(f"TransformerAngleVectorEmbedding does not support {type_name(model)}")

    def make_embedder(self) -> Model:
        "Creats the keras model for the embedding."

        assert self.n_embd % 2 == 0, f"n_embd must be divisible by 2 to use angle embedding, got n_embd={self.n_embd}"
        assert self.task_cfg is not None, "Must call task.configure(embedding) before embedding.make_embedder()."

        pos_embedder = tf.keras.layers.Embedding(self.task_cfg.sequence_length, self.n_embd, name="pos_embedder")
        dense_out = tf.keras.layers.Dense(self.n_embd, name="embd")

        import mx.layers as mxl
        prepend_begin_token = mxl.prepend_token(
            token=mxl.tokens.BEGIN,
            n_embd=self.n_embd,
        )

        def embed(inputs):

            ## make angle embeddings
            angles = inputs["angles"]
            scale = (tau / 4.) * (1. / self.n_repeats) # only need to produce rotations up to tau/4, because the model can easily invert angles
            offsets = tf.range(self.n_repeats, dtype=tf.float32) * scale
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
            Input([None, self.task_cfg.n_input_dims], dtype=tf.float32, name="angles"),
            Input([None],                             dtype=tf.int32,   name="seq_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name="TransformerAngleVectorEmbedding",
        )
