import functools
from math import pi, tau
from typing import Literal
from typing_extensions import Self

import tensorflow as tf
import typing
if typing.TYPE_CHECKING:
    import keras.api._v2.keras as keras
    from keras.api._v2.keras import Model, Input, layers
    from keras.api._v2.keras.backend import is_keras_tensor
    from tensorflow.python.types.core import TensorLike
else:
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers
    from tensorflow.keras.backend import is_keras_tensor
    from tensorflow.types.experimental import TensorLike
# these two imports are actually from tensorflow.python, not just for type checking
from tensorflow.python.util import tf_decorator
from tensorflow.python.module.module import camel_to_snake

class Einshape:

    def __init__(self, batch_dims: dict[str, int | None], sequence_dims: dict[str, int | None | Literal["ragged"]], feature_dims: dict[str, int | None]):
        self._b = batch_dims
        self._s = {
            k: None if v == "ragged" else v
            for k, v in sequence_dims.items()
        }
        self._is_ragged = {
            k: v == "ragged"
            for k, v in sequence_dims.items()
        }
        self._f = feature_dims
    
    def is_ragged(self, key: str) -> bool:
        return self._is_ragged[key]

    @property
    def b(self) -> dict[str, int | None]:
        """Batch dimensions."""
        return self._b
    @property
    def s(self) -> dict[str, int | None]:
        """Sequence dimensions."""
        return self._s
    @property
    def f(self) -> dict[str, int | None]:
        """Feature dimensions."""
        return self._f

    @property
    def b_str(self) -> str:
        return " ".join(k for k in self._b.keys())
    @property
    def s_str(self) -> str:
        return " ".join(k for k in self._s.keys())
    @property
    def f_str(self) -> str:
        return " ".join(k for k in self._f.keys())

    @property
    def shape(self):
        """Return the shape of the tensor as a list of integers or None."""
        return [*self._b.values(), *self._s.values(), *self._f.values()]

    def b_s_shape(self) -> list[int | None]:
        """Shape of the batch and sequence dimensions."""
        return [*self._b.values(), *self._s.values()]

    @property
    def s_f_shape(self) -> list[int | None]:
        """Shape of the sequence and feature dimensions."""
        return [*self._s.values(), *self._f.values()]

    @property
    def b_shape(self) -> list[int | None]:
        return [dim for dim in self._b.values()]
    @property
    def s_shape(self) -> list[int | None]:
        return [dim for dim in self._s.values()]
    @property
    def f_shape(self) -> list[int | None]:
        return [dim for dim in self._f.values()]

    @property
    def rank(self) -> int:
        return len(self._b) + len(self._s) + len(self._f)
    
    @property
    def b_rank(self) -> int:
        return len(self._b)
    @property
    def s_rank(self) -> int:
        return len(self._s)
    @property
    def f_rank(self) -> int:
        return len(self._f)

    @staticmethod
    def _product(shape: list[int | None]) -> int:
        
        def multiply_or_none(x, y):
            if x is None or y is None:
                return None
            else:
                return x * y

        return functools.reduce(multiply_or_none, shape)

    @property
    def b_product(self) -> int:
        """Return the total length of the batch dimensions (product of all batch dimensions)."""
        return Einshape._product(self.b_shape)

    @property
    def s_product(self) -> int:
        """Return the total length of the sequence dimensions (product of all sequence dimensions)."""
        return Einshape._product(self.s_shape)

    @property
    def f_product(self) -> int:
        """Return the total length of the feature dimensions (product of all feature dimensions)."""
        return Einshape._product(self.f_shape)
    
    @property
    def product(self) -> int:
        """Return the total length of the tensor (product of all dimensions)."""
        return Einshape._product(self.shape)

    def cut(self, new_seq_dims: list[int | None]) -> Self:
        """Cut the sequence dimensions to the given lengths. New sequence dimensions must be shorter than the old ones."""

        assert len(new_seq_dims) == self.s_rank, f"Expected {self.s_rank} sequence dimensions, got {len(new_seq_dims)}."
        assert all(dim is None or dim > 0 for dim in new_seq_dims), "Sequence dimensions must be positive integers."
        assert all(dim is None or dim <= old_dim for dim, old_dim in zip(new_seq_dims, self.s_shape)), "New sequence dimensions must be smaller than old sequence dimensions."

        return Einshape(
            batch_dims = self._b,
            sequence_dims = { k: dim for k, dim in zip(self._s.keys(), new_seq_dims) },
            feature_dims = self._f,
        )
    
    def project(self, new_feature_dims: list[int | None]) -> Self:
        """Project the feature dimensions to the given lengths."""

        assert len(new_feature_dims) == self.f_rank, f"Expected {self.f_rank} feature dimensions, got {len(new_feature_dims)}."
        assert all(dim is None or dim > 0 for dim in new_feature_dims), "Feature dimensions must be positive integers."

        return Einshape(
            batch_dims = self._b,
            sequence_dims = self._s,
            feature_dims = { k: dim for k, dim in zip(self._f.keys(), new_feature_dims) },
        )

    def f_indices(self, flatten=True, elide_rank_1=True):
        """
        Return a list of indices for the feature dimensions.
        
        If flatten=True (the default), the returned indices will
        have shape [ product(f_shape), f_rank]. Otherwise, the
        returned indices will have shape [ *f_shape, f_rank ].
        
        If elide_rank_1=True (the default), when there is only a
        single feature dimension, the returned indices will not have
        an extra dimension. Otherwise, the returned indices will have
        an extra dimension with size equal to the rank of the feature
        dimensions.
        """

        return multidim_indices(self.f_shape, flatten=flatten, elide_rank_1=elide_rank_1)

    def s_indices(self, flatten=True, elide_rank_1=True):
        """
        Return a list of indices for the sequence dimensions.
        
        If flatten=True (the default), the returned indices will
        have shape [ s_product, s_rank]. Otherwise, the
        returned indices will have shape [ *s_shape, s_rank ].
        
        If elide_rank_1=True (the default), when there is only a
        single sequence dimension, the returned indices will not have
        an extra dimension. Otherwise, the returned indices will have
        an extra dimension with size equal to the rank of the sequence
        dimensions.
        """

        return multidim_indices(self.s_shape, flatten=flatten, elide_rank_1=elide_rank_1)
    
    def b_indices(self, flatten=True, elide_rank_1=True):
        """
        Return a list of indices for the batch dimensions.

        If flatten=True (the default), the returned indices will
        have shape [ b_product, b_rank]. Otherwise, the returned
        indices will have shape [ *b_shape, b_rank ].

        If elide_rank_1=True (the default), when there is only a
        single batch dimension, the returned indices will not have
        an extra dimension. Otherwise, the returned indices will have
        an extra dimension with size equal to the rank of the batch
        dimensions.
        """

        return multidim_indices(self.b_shape, flatten=flatten, elide_rank_1=elide_rank_1)
    
    def indices(self, flatten=True):
        """
        Return a list of indices for the batch, sequence, and feature dimensions.
        
        If flatten=True (the default), the returned indices will
        have shape [ product, rank ]. Otherwise, the
        returned indices will have shape [ *shape, rank ].
        """
        return multidim_indices(self.shape, flatten=flatten, elide_rank_1=False)

    def append_feature_dim(self, name: str, val: int | None) -> Self:
        """Append a new feature dimension to the shape."""
        assert name not in self._f, f"Feature dimension {name} already exists."
        return Einshape(
            batch_dims = self._b,
            sequence_dims = self._s,
            feature_dims = { **self._f, name: val },
        )

    def with_feature_dims(self, feature_dims: dict[str, int | None]) -> Self:
        """Return a new shape with the given feature dimensions."""
        return Einshape(
            batch_dims = self._b,
            sequence_dims = self._s,
            feature_dims = feature_dims,
        )
    
    def with_sequence_dims(self, sequence_dims: dict[str, int | None | Literal["ragged"]]) -> Self:
        """Return a new shape with the given sequence dimensions."""
        return Einshape(
            batch_dims = self._b,
            sequence_dims = sequence_dims,
            feature_dims = self._f,
        )
    
    def with_batch_dims(self, batch_dims: dict[str, int | None]) -> Self:
        """Return a new shape with the given batch dimensions."""
        return Einshape(
            batch_dims = batch_dims,
            sequence_dims = self._s,
            feature_dims = self._f,
        )
    


def multidim_indices(shape, flatten=True, elide_rank_1=True):
    """
    Uses tf.meshgrid to get the indices for a tensor of any rank
    Returns an int32 tensor of shape [ product(shape), rank ]
    """
    if len(shape) == 0:
        raise ValueError("Shape must have at least one dimension.")
    if len(shape) == 1:
        if elide_rank_1:
            return tf.range(shape[0], dtype=tf.int32)

    indices = tf.meshgrid(*[tf.range(s) for s in shape])
    indices = tf.stack(indices, axis=-1)
    if flatten:
        indices = tf.reshape(indices, [-1, len(shape)])
    return indices


def multidim_indices_of(tensor, flatten=True):
    """
    Uses tf.meshgrid to get the indices for a tensor of any rank
    Returns an int32 tensor of shape [ product(shape), rank ]
    """
    shape = tf.shape(tensor)
    return multidim_indices(shape, flatten=flatten)


def angle_wrap(angles):
    """
    Wrap angle in radians to [-pi, pi] range
    """
    angles = (angles + pi) % tau - pi
    # angles = tf.math.atan2(tf.sin(angles), tf.cos(angles))
    return angles


def circular_mean(angles, axis=0):
    # compute the circular mean of the data for this example+track
    # rotate the data so that the circular mean is 0
    # store the circular mean
    means_cos_a = tf.reduce_mean(tf.math.cos(angles), axis=axis)
    means_sin_a = tf.reduce_mean(tf.math.sin(angles), axis=axis)
    circular_means = tf.math.atan2(means_sin_a, means_cos_a)
    return circular_means


def recluster(angles, circular_means=None, frame_axis=0):
    if circular_means is None:
        circular_means = circular_mean(angles, axis=frame_axis)

    # rotate the data so the circular mean is 0
    angles = angles - tf.expand_dims(circular_means, axis=frame_axis)
    angles = angle_wrap(angles)

    return angles


def unrecluster(angles, circular_means, n_batch_dims=0):
    # assuming the mean is currently 0, rotate the data so the mean is
    # back to the original given by `circular_means`
    circular_means = tf.expand_dims(circular_means, axis=0) # add frame axis
    for _ in range(n_batch_dims):
        circular_means = tf.expand_dims(circular_means, axis=0) # add batch_dims
    angles = angles + circular_means
    angles = angle_wrap(angles)

    return angles

def tf_scope(func):
    """
    Decorator to automatically enter the module name scope.

    This will create a scope named after:
    -   The module name (if the wrapped function is __call__)
    -   The module name + "_init" (if the wrapped function is __init__)
    -   Any `name` argument passed to the wrapped function
    -   The function name (otherwise)

    """

    is_init = func.__name__ == "__init__"
    is_call = func.__name__ == "__call__"
    
    fn_name = func.__name__

    @functools.wraps(func)
    def func_with_name_scope(*args, **kwargs):
        is_module = len(args) > 0 and isinstance(args[0], tf.Module)
        
        if is_module and is_init:
            # init happens before the tf.Module instance has a _name attribute
            if 'name' in kwargs and kwargs['name'] is not None:
                name_prefix = kwargs['name']
            else:
                name_prefix = camel_to_snake(type(args[0]).__name__)
            
            scope_name = name_prefix + "_init"
        elif is_module and not is_call:
            scope_name = args[0].name + "_" + fn_name
        elif is_module and is_call:
            scope_name = args[0].name
        else:
            scope_name = fn_name
        
        with tf.name_scope(scope_name):
            return func(*args, **kwargs)
    
    return tf_decorator.make_decorator(func, func_with_name_scope)
