import abc
from dataclasses import dataclass
from typing import Literal, Union

@dataclass
class TaskCfg(abc.ABC):
    """Defines a particular input and output format that a dataset can be adapted into."""

    batch: Union[Literal[False], int]
    """
    Whether the task includes batching the data, and if so, the batch size.
    """

@dataclass
class NextTokenPrediction(TaskCfg):
    """Next token prediction task."""

    sequence_length: int
    """
    The sequence length used when training. This
    is the maximum length that the model will be
    able to predict reliably, unless
    'relative_only' is used.
    """

    batch = False

    relative_only: bool = False
    """
    If True, the model will only be able to predict
    the next token using the relative position of
    the context. This is useful for very long
    sequences, and means the model will generalize
    any length of sequence.
    """

@dataclass
class MaskedSequenceModeling(TaskCfg):
    """Fill-in-the-gaps task."""

    sequence_length: int
    """
    The sequence length used when training. This
    is the maximum length that the model will be
    able to predict reliably, unless
    'relative_only' is used.
    """

    mask_probability: float = 0.15
    """
    The probability that a token will be masked.
    """

    relative_only: bool = False
    """
    If True, the model will only be able to predict
    the next token using the relative position of
    the context. This is useful for very long
    sequences, and means the model will generalize
    any length of sequence.
    """


@dataclass
class QueryPrediction(TaskCfg):
    """Predict new tokens at specified positions."""

    sequence_length: int
    """
    The sequence length used when training. This
    is the maximum length that the model will be
    able to predict reliably, unless
    'relative_only' is used.
    """

    max_distance: int
    """
    The maximum distance between a query token and the context
    during training. This is the maximum distance at which the
    model will be able to predict reliably.
    """

    n_queries: int
    """
    The number of query tokens to predict during training.
    """

    n_context_tokens: int
    """
    The number of tokens to use as context during training.
    This is the maximum number of tokens that the model
    will be able to use reliably.
    """
    
    relative_only: bool = False
    """
    If True, the model will only be able to predict
    the next token using the relative position of
    the context. This is useful for very long
    sequences, and means the model will generalize
    any length of sequence.
    """


TaskCfg = Union[
    NextTokenPrediction,
    MaskedSequenceModeling,
    QueryPrediction,
]
