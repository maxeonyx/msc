import abc
from dataclasses import KW_ONLY, dataclass
from typing import Callable, Generic, Type, TypeVar

from mx.datasets import Dataset, DSets
from mx.models import MxModel

from mx.utils.tf import *
from mx.layers import input_dict

EmbeddingToModelAdaptor = Callable[[MxModel], tf.keras.layers.Layer]

class Embedding(abc.ABC):
    """
    Base class for embeddings.
    """

    @property
    @staticmethod
    @abc.abstractmethod
    def Config() -> Type:
        pass

    def __init__(self, name: str, identifier: str, n_embd: int) -> None:
        self.name = name
        "Human-readable name"

        self.identifier = identifier
        "Unique, machine-readable identifier"

        self.n_embd = n_embd
        "Number of embedding dimensions"

    @property
    def embedding_spec(self) -> dict[str, int]:
        pass

    @property
    @abc.abstractmethod
    def implementations(self) -> dict:
        pass

    def is_compatible_with(self, model: MxModel) -> bool:
        return type(model) in self.implementations
    
    def build(self, model: MxModel, data: DSets) -> tf.keras.layers.Layer:
        if type(model) not in self.implementations:
            raise NotImplementedError(f"Model {model.name} not implemented for embedding {self.name}")
        else:
            adaptor = self.implementations[type(model)]

        return adaptor(data)



class TransformerVectorEmbedding(Embedding):
    """
    Simple embedding for transformer regression. Codebook embeddings for positions,
    and linear embeddings for inputs.
    """

    @dataclass
    class Config:

        sequence_length: int
        """
        Max length of the sequence to be embedded.
        Max value among seq_idxs.
        """

        n_input_dims: int
        """
        Size of the input vector.
        """
    
    def __init__(self, name: str, identifier: str) -> None:
        super().__init__(
            name,
            identifier
        )

    @property
    def implementations(self) -> dict:
        return {
            
        }

    def build(self, cfg: Config) -> Model:
        
        input_embedder = tf.keras.layers.Dense(cfg.n_embd, name="input_embedder")
        pos_embedder = tf.keras.layers.Embedding(cfg.sequence_length, cfg.n_embd, name="pos_embedder")
        def embed(inputs):
            input = inputs["input"]
            input_idxs = inputs["input_idxs"]
            return input_embedder(input) + pos_embedder(input_idxs)
        inputs = input_dict(
            Input([None, cfg.n_input_dims], dtype=tf.float32, name="input"),
            Input([None],                   dtype=tf.int32,   name="input_idxs"),
        )
        return Model(
            inputs=inputs,
            outputs=embed(inputs),
            name="TransformerVectorEmbedding",
        )


class TransformerAngleVectorEmbedding(Embedding):
    """
    Simple embedding for transformer regression over angles.

    Embeds angles as unit vectors. Creates `n_repeats` copies of them, rotated
    evenly around one quarter of the unit circle.

    Embeds positions with a codebook.
    """

    @dataclass
    class Config:

        sequence_length: int
        """
        Max length of the sequence to be embedded.
        Max value among seq_idxs.
        """

        n_input_dims: int
        """
        Size of the input vector.
        """
    
    def __init__(self, name: str, identifier: str, n_repeats: int) -> None:
        super().__init__(
            name,
            identifier,
        )
        self.n_repeats = n_repeats
        "Number of additional rotations of each angle to be added to the embedding."

    def implementations(self) -> dict:
        return {
        }

    def _make_layer(self, ctx: Config) -> Model:
        

        assert ctx.n_embd % 2 == 0, f"n_embd must be divisible by 2 to use angle embedding, got n_embd={ctx.n_embd}"

        dense_out = tf.keras.layers.Dense(ctx.n_embd, name="embd")

        def angle_call(angles):
            scale = (tau / 4.) * (1. / self.n_repeats) # only need to produce rotations up to tau/4, because the model can easily invert angles
            offsets = tf.range(self.n_repeats, dtype=tf.float32) * scale
            # add "repeats" dim
            angles = angles[..., None]
            angles = angles + tf.broadcast_to(offsets, tf.broadcast_dynamic_shape(tf.shape(offsets), tf.shape(angles)))
            # add "sincos" dim
            angles = tf.stack([tf.sin(angles), tf.cos(angles)], axis=-1)
            # flatten to "embd" dim
            embd = dense_out(angles)

            return embd

        inputs = input_dict(
            Input(shape=[None, ctx.n_input_dims], dtype=tf.float32, name="angles"),
            Input(shape=[None],                   dtype=tf.int32,   name="input_idxs"),
        )
        
        return Model(inputs=inputs, outputs=angle_call(**inputs), name="embd")
