from abc import abstractmethod, ABC
import inspect
from collections import UserDict
from collections.abc import Callable
import functools
from math import pi, tau
from random import random
from typing_extensions import Self
from typing import Literal, Callable, ParamSpec, Union

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

def soft_convert_to_tensor_recursive(d):
    """
    Recursively converts all dicts in a nested dict structure to tensors.
    Converts tuples recursively, but converts any lists to tensors.
    """
    if isinstance(d, Einsor):
        return d
    
    if isinstance(d, tuple):
        return tuple(soft_convert_to_tensor_recursive(x) for x in d)
    
    if (
        isinstance(d, list)
        or isinstance(d, int)
        or isinstance(d, float)
        or isinstance(d, str)
        or isinstance(d, bool)
    ):
        print("Converting to tensor:", d)
        print("Converting to tensor:", d)
        print("Converting to tensor:", d)
        print("Converting to tensor:", d)
        print("Converting to tensor:", d)
        print("Converting to tensor:", d)
        print("Converting to tensor:", d)
        print("Converting to tensor:", d)
        return tf.convert_to_tensor(d)

    if (
        isinstance(d, dict)
        or isinstance(d, UserDict)
    ):
        if isinstance(d, UserDict):
            d = d.data
        for k, v in d.items():
            d[k] = soft_convert_to_tensor_recursive(v)
        return d
    
    return d


class Einsor(tf.Tensor, UserDict):
    """
    A TensorLike object which can have a default value, as well as an auxiliary
    dictionary of values.
    """
    val: tf.Tensor
    extra: typing.Mapping[str, tf.Tensor]

    def __init__(self, val: TensorLike, **extra: typing.Mapping[str, tf.Tensor]):
        self.val = soft_convert_to_tensor_recursive(val)
        self.extra = soft_convert_to_tensor_recursive(extra)

    @staticmethod
    def _proxy_methods() -> set[str]:
        return {
            "numpy",
            "shape",
            "dtype",
            "ndim",
            "__iter__",
            *tf.Tensor.OVERLOADABLE_OPERATORS,
        }
    
    @staticmethod
    def _make_proxy_method(name):
        def method(self, *args, **kwargs):
            print("Calling method:", name)
            return getattr(self.val, name)(*args, **kwargs)
        return method
    
    @classmethod
    def _attach_proxy_methods(cls):
        for method_name in Einsor._proxy_methods():
            method = cls._make_proxy_method(method_name)
            setattr(cls, method_name, method)

    def __getitem__(self, key: str) -> tf.Tensor:
        if type(key) is str:
            return self.extra[key]
        else:
            return self.val[key]

Einsor._attach_proxy_methods()



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
        return [*self._b.values(), *self._s.values(), *self._f.values()]

    @property
    def s_f_shape(self) -> list[int | None]:
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


def multidim_indices(shape, flatten=True):
    """
    Uses tf.meshgrid to get the indices for a tensor of any rank
    Returns an int32 tensor of shape [ product(shape), rank ]
    """
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


class Layer(Callable, tf.Module, ABC):
    pass


class Layer_DefaultInput(Layer, ABC):

    @abstractmethod
    def __call__(self, input: Einsor, *args, **kwargs) -> Union[Einsor, dict[str, Einsor]]:
        pass

    @classmethod
    def get_all_subclasses(cls):
        all_subclasses = []

        for subclass in cls.__subclasses__():
            if not inspect.isabstract(subclass):
                all_subclasses.append(subclass)
            all_subclasses.extend(cls.get_all_subclasses(subclass))

        return all_subclasses

class Layer_DefaultOutput(Layer, ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Einsor:
        pass
