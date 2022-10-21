from __future__ import annotations
from contextlib import contextmanager
import functools
import itertools
from typing import Dict

from box import Box

# these two imports are actually from tensorflow.python, not just for type checking
from tensorflow.python.util import tf_decorator
from tensorflow.python.module.module import camel_to_snake

from mx.prelude import *

__all__, export = exporter()

from mx.run_name import get_run_name, random_run_name, set_run_name
export("get_run_name")
export("random_run_name")
export("set_run_name")

@export
def dtype() -> tft.DType:
    policy = keras.mixed_precision.global_policy()
    return policy.compute_dtype

@export
def reg():
    return tf.keras.regularizers.l1_l2(l1=0.0001, l2=0.001)

@export
@contextmanager
def optimizations(options):
  old_opts = tf.config.optimizer.get_experimental_options()
  tf.config.optimizer.set_experimental_options(options)
  try:
    yield
  finally:
    tf.config.optimizer.set_experimental_options(old_opts)

@export
def validate(x, var_name, spec, part="", allow_non_concrete=False):
    """
    Validate that the input is a tensor-like nested structure in the correct format.

    Format should be a nested structure of dicts, lists and tuples with TensorSpecs and
    NoneTensorSpecs at the nodes.

    Lists used for variable-length homogeneous structures, and the format spec should only have 1 element.
    For fixed-length / heterogeneous structures, use tuples.
    """
    name = f"{var_name}{part}"
    # Use type_name and tf_str to print pretty error messages.

    got_val = lambda x, name: f"Got: {name} = {tf_str(x)}"
    got_type = lambda x, name: f"Got: type({name}) = {type_name(x)}"

    if isinstance(spec, (tf.TensorSpec, tf.RaggedTensorSpec)):
        if not allow_non_concrete and not tf.is_tensor(x):
            raise TypeError(f"Expected {name} to be a tensor. {got_type(x, name)}")

        if not spec.is_compatible_with(x):
            raise TypeError(f"\nExpected: {name} ∈ {tf_str(spec)}\nGot:      {name} = {tf_str(x)}")

    elif isinstance(spec, tft.NoneTensorSpec):
        if not spec.is_compatible_with(x):
            raise TypeError(f"Expected {name} to be None. {got_val(x, name)}")

    elif isinstance(spec, tft.DatasetSpec):
        validate(x.element_spec, var_name, spec.element_spec, '.element_spec', allow_non_concrete=True)
    elif isinstance(spec, dict):
        if not isinstance(x, dict):
            raise TypeError(f"Expected {name} to be a dict. {got_type(x, name)}")

        for k, s in spec.items():

            if k not in x:
                raise KeyError(f"Expected {name} to have key {k}. {got_val(x, name)}")

            validate(x[k], var_name, s, f"[{k!r}]", allow_non_concrete)

    # list specs should only have 1 element
    elif isinstance(spec, list):
        if not isinstance(x, list):
            raise TypeError(f"Expected {name} to be a list. {got_type(x, name)}")

        if len(spec) != 1:
            raise ValueError(f"`list` specs should only have 1 element. Use tuples instead. Got: spec{part} = {tf_str(spec)}")

        for i, v in enumerate(x):
            validate(v, var_name, spec[0], f"[{i}]", allow_non_concrete)

    # tuple specs can have multiple elements
    elif isinstance(spec, tuple):
        if not isinstance(x, tuple):
            raise TypeError(f"Expected {name} to be a tuple. {got_type(x, name)}")

        if len(x) != len(spec):
            raise ValueError(f"Expected {name} to have {len(spec)} elements. Got len({name}) = {len(x)}")

        for i, (v, s) in enumerate(zip(x, spec)):
            validate(v, var_name, s, f"[{i}]", allow_non_concrete)

    else:
        raise ValueError(f"Invalid spec. Expected TensorSpec, NoneTensorSpec, dict, list or tuple. Got spec{part} = {repr(spec)}")

@export
def funky_punky(n_0, n_max, n_total, base=2):
    """
    Yields from a funky-punky series, such that the sum of the series is `n`.

    >>> list(funky_punky(5, 99, 10))
    [5, 5]
    >>> list(funky_punky(5, 99, 200))
    [5, 8, 16, 32, 64, 75]
    >>> sum(funky_punky(5, 99, 200)) == 200
    True
    >>> list(funky_punky(5, 30, 200))
    [5, 8, 16, 30, 30, 30, 30, 30, 21]
    """
    if n_0 > n_total:
        yield n_0 - n_total
        return
    yield n_0
    total = n_0
    i = 0
    while base**i < n_0:
        i += 1

    while True:
        if base ** i > n_max:
            n = n_max
        else:
            n = base ** i
            i += 1
        if total + n > n_total:
            break
        yield n
        total += n
    yield n_total - total


@export
def exponential_up_to(n, base=2):
    """
    Yields from an exponential series, such that the sum of the series is `n`.

    >>> list(exponential_up_to(10))
    [1, 2, 4, 3]
    >>> list(exponential_up_to(10, base=3))
    [1, 3, 6]
    """
    i = 0
    total = 0
    while True:
        e = base ** i
        if total + e > n:
            break
        yield e
        i += 1
        total += e
    yield n - total

@export
def constant_up_to(n, chunk):
    """
    Yields from a constant series, such that the sum of the series is `n`.

    >>> list(constant_up_to(10, 2))
    [2, 2, 2, 2, 2]
    >>> list(constant_up_to(10, 3))
    [3, 3, 3, 1]
    """
    total = 0
    while True:
        if total + chunk > n:
            break
        yield chunk
        total += chunk

    if total < n:
        yield n - total


_default_indent = "  "
@export
def set_default_indent(indent: int | str):
    global _default_indent
    if isinstance(indent, int):
        _default_indent = " " * indent
    else:
        _default_indent = indent


debug = True
@export
def set_debug(to: bool = True):
    global debug
    debug = to

# Run a keras model up to a specific layer
@export
def stats(val, indent=_default_indent, depth=0):
    def stats(v):
        print(tf_str(val, prefix="val: ", indent=indent, depth=depth))
        print(indent*(depth+1) + "mean:", tf.math.reduce_mean(v).numpy())
        print(indent*(depth+1) + "min:", tf.math.reduce_min(v).numpy())
        print(indent*(depth+1) + "max:", tf.math.reduce_max(v).numpy())
        if v.dtype in [tf.float16, tf.float32, tf.float64]:
            print(indent*(depth+1) + "norm:", tf.linalg.norm(v).numpy())
            print(indent*(depth+1) + "std:", tf.math.reduce_std(v).numpy())
            print(indent*(depth+1) + "is NaN?", tf.math.reduce_any(tf.math.is_nan(v)).numpy())
    tf.nest.map_structure(stats, val)

@export
def mo_all(x, model, indent=_default_indent, depth=0):
    print(indent*depth + model.name)
    stats(model(x), indent=indent, depth=depth+1)
    print()
    val = x
    for i in range(len(model.layers)):
        l = model.layers[i]
        try:
            val = l(val)
            if hasattr(l, "layers") and len(l.layers) > 0:
                mo_all(val, l, depth=depth+1, indent=indent)
            else:
                print(indent*depth + l.name)
                stats(val, depth=depth+1, indent=indent)
                print()
        except Exception as e:
            print(indent*depth + l.name)
            print(indent*(depth+1) + f"Error computing output for layer {i}: {l.name}")
            print(indent*(depth+1) + str(e)[:100])
            print()
    print(indent*depth + model.name)
    stats(model(x), indent=indent, depth=depth+1)
    print()


@export
def is_debug():
    return debug

def primes():
    """
    Generate an infinite sequence of prime numbers.

    >>> list(itertools.islice(primes(), 10))
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    """
    # https://stackoverflow.com/a/568618/123879
    D = {}
    q = 2

    while True:
        if q not in D:
            yield q
            D[q * q] = [q]
        else:
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]
        q += 1

def debug_numbers():
    p = iter(primes)
    def next_prime_or(val, multiple_of=1):
        if debug:
            return next(p)*multiple_of
        else:
            return val
    return next_prime_or


def next_prime_after(n):
    """
    Returns the very next prime number after `n`.

    >>> next_prime_after(10)
    11
    >>> next_prime_after(11)
    11
    >>> next_prime_after(20)
    23
    """
    for p in primes():
        if p >= n:
            return p


def list_to_dict(l):
    """
    Turns a list of objs with `name` attributes into a dict of name -> obj.

    >>> class Foo:
    ...     def __init__(self, name):
    ...         self.name = name
    ...     def __repr__(self):
    ...         return f"Foo({self.name})"
    >>> list_to_dict([Foo("a"), Foo("b")])
    {'a': Foo(a), 'b': Foo(b)}
    """
    return { x.name: x for x in l }

def shape(tensor: typing.Union[tf.Tensor, np.ndarray]) -> list[int | tf.Tensor]:
    """
    Deal with dynamic shape "cleanly".

    Returns:
        `List[int | tf.Tensor]`: The shape of the tensor as a list. If a
        particular dimension is only known dynamically, it will be a scalar
        `tf.Tensor` object.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

@export
def input_dict(*arr):
    return list_to_dict(arr)

@export
def type_name(x):
    return type(x).__name__

@export
class DSets(Box):

    def __init__(
        self,
        train: Dataset,
        test: Dataset,
        val: Dataset,
    ):
        self.train = train
        self.test = test
        self.val = val

    def destructure(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        return self.train, self.test, self.val

    def map(self, fn) -> DSets:
        return DSets(
            # train = self.train.map(fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
            # test = self.test.map(fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
            # val = self.val.map(fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
            train = self.train.map(fn),
            test = self.test.map(fn),
            val = self.val.map(fn),
        )

    def batch(self, batch_size, test_batch_size) -> DSets:
        dset = DSets(
            # train = self.train.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
            # test = self.test.batch(test_batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
            # val = self.val.batch(test_batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
            train = self.train.batch(batch_size),
            test = self.test.batch(test_batch_size),
            val = self.val.batch(test_batch_size),
        )

        return dset

    def cache(self) -> DSets:
        return DSets(
            train = self.train.cache(),
            test = self.test.cache(),
            val = self.val.cache(),
        )

@export
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

@export
class Einshape:

    def __init__(self, batch_dims: dict[str, int | None] = {}, sequence_dims: dict[str, int | None | Literal["ragged"]] = {}, feature_dims: dict[str, int | None] = {}):
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
    def b(self) -> int | None:
        """Batch size."""
        assert len(self._b) == 1, f"Einshape.b can only be used if there is exactly one batch dimension. Got {self.b_dict}."
        return self._b.values()[0]

    @property
    def s(self) -> dict[str, int | None]:
        """Sequence length."""
        assert len(self._s) == 1, f"Einshape.s can only be used if there is exactly one sequence dimension. Got {self.s_dict}."
        return self._s.values()[0]

    @property
    def f(self) -> dict[str, int | None]:
        """Feature dimensions."""
        assert len(self._f) == 1, f"Einshape.f can only be used if there is exactly one feature dimension. Got {self.f_dict}."
        return self._f.values()[0]


    @property
    def b_dict(self) -> dict[str, int | None]:
        """Batch dimensions."""
        return self._b
    @property
    def s_dict(self) -> dict[str, int | None]:
        """Sequence dimensions."""
        return self._s
    @property
    def f_dict(self) -> dict[str, int | None]:
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

    @property
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

    def cut(self, new_seq_dims: list[int]) -> Self:
        """Cut the sequence dimensions to the given lengths. New sequence dimensions must be shorter than the old ones."""

        assert len(new_seq_dims) == self.s_rank, f"Expected {self.s_rank} sequence dimensions, got {len(new_seq_dims)}."
        assert all(dim > 0 for dim in new_seq_dims), "Sequence dimensions must be positive integers."
        assert all(old_dim is None or dim <= old_dim for dim, old_dim in zip(new_seq_dims, self.s_shape)), "New sequence dimensions must be smaller than old sequence dimensions."

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

    def batch(self, new_batch_dim: int, name="b") -> Self:
        """Prepends a new batch dimension to the tensor."""

        assert isinstance(new_batch_dim, int) and new_batch_dim > 0, "Batch dimension must be a positive integer."

        return self.with_batch_dims({ name: new_batch_dim, **self.b_dict, })

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

@export
@tf_scope
def multidim_indices(shpe, flatten=True, elide_rank_1=True):
    """
    Uses tf.meshgrid to get multidimensional indices in the given shape.

    If flatten=True (the default), the returned indices will
    have shape [ product(shape), rank]. Otherwise, the
    returned indices will have shape [ *shape, rank ].

    If elide_rank_1=True (the default), when there is only a
    single dimension, the returned indices will not have
    an extra dimension. Otherwise, the returned indices will have
    an extra dimension with size equal to the rank of the tensor.
    """

    if len(shpe) == 0:
        raise ValueError("Shape must have at least one dimension.")
    if len(shpe) == 1 and elide_rank_1:
        return tf.range(shpe[0], dtype=tf.int32)

    indices = tf.meshgrid(*[tf.range(s) for s in shpe], indexing="ij")
    indices = tf.stack(indices, axis=-1)
    if flatten:
        indices = tf.reshape(indices, [-1, len(shpe)])
    return indices

@export
@tf_scope
def multidim_indices_of(tensor, flatten=True, elide_rank_1=True):
    """
    Uses tf.meshgrid to get multidimensional indices in the shape of the given tensor.

    If flatten=True (the default), the returned indices will
    have shape [ product(shape), rank]. Otherwise, the
    returned indices will have shape [ *shape, rank ].

    If elide_rank_1=True (the default), when there is only a
    single dimension, the returned indices will not have
    an extra dimension. Otherwise, the returned indices will have
    an extra dimension with size equal to the rank of the tensor.
    """

    s = shape(tensor)
    return multidim_indices(s, flatten=flatten, elide_rank_1=elide_rank_1)

@export
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

@export
@tf_scope
def recluster(angles, circular_means=None, frame_axis=0):
    if circular_means is None:
        circular_means = circular_mean(angles, axis=frame_axis)

    # rotate the data so the circular mean is 0
    angles = angles - tf.expand_dims(circular_means, axis=frame_axis)
    angles = angle_wrap(angles)

    return angles

@export
@tf_scope
def unrecluster(angles, circular_means, n_batch_dims=0):
    # assuming the mean is currently 0, rotate the data so the mean is
    # back to the original given by `circular_means`
    circular_means = tf.expand_dims(circular_means, axis=0) # add frame axis
    for _ in range(n_batch_dims):
        circular_means = tf.expand_dims(circular_means, axis=0) # add batch_dims
    angles = angles + circular_means
    angles = angle_wrap(angles)

    return angles

def tf_repr(x, indent=_default_indent, depth=0, prefix=""):

    # container types
    if isinstance(x, dict):
        return tf_dict_repr(x, indent=indent, depth=depth, prefix=prefix)
    elif isinstance(x, tuple):
        return tf_tuple_repr(x, indent=indent, depth=depth, prefix=prefix)
    elif isinstance(x, list):
        return tf_list_repr(x, indent=indent, depth=depth, prefix=prefix)

    elif isinstance(x, Dataset) or isinstance(x, tft.DatasetSpec):

        if isinstance(x, Dataset):
            mark = "●"
            cardinality = x.cardinality()
            if cardinality == tf.data.INFINITE_CARDINALITY:
                cardinality = "∞"
            else:
                cardinality = tf.strings.as_string(cardinality)
        else:
            mark = "○"
            cardinality = "?"

        prefix = tf.strings.join([
            prefix,
            mark,
            "Dataset[",
            cardinality,
            "] ",
        ])
        return tf_repr(x.element_spec, indent=indent, depth=depth, prefix=prefix)
    elif isinstance(x, tf.TensorSpec):
        x.shape
        c = "○" # ○ means it's not concrete
        if x.shape.rank == 0:
            str_x = tf.constant(f"{c}{x.dtype.name}[]")
        else:
            str_x = tf.strings.join([
                c,
                x.dtype.name,
                tf_shape_repr(x.shape),
            ])
    elif isinstance(x, tft.NoneTensorSpec):
        str_x = tf.constant("○None")
    elif isinstance(x, tft.NoneTensor):
        str_x = tf.constant("●None")
    elif x is None:
        str_x = tf.constant("None")
    elif not tf.is_tensor(x):
        str_x = tf.constant(repr(x), tf.string)
    elif x.shape.rank == 0:
        if x.dtype == tf.string:
            str_x = tf.strings.join(["●\"", x, "\""])
        else:
            str_x = tf.strings.as_string(x)
    else:
        str_x = tf.strings.join([
            "●", # ● means concrete
            x.dtype.name,
            tf_shape_repr(shape(x)),
        ])
    str_x

    return tf.strings.join([
        indent*depth,
        prefix,
        str_x,
    ])

def tf_shape_repr(x):
    """
    The [1 2 3] part of e.g. !int32[1 2 3]
    """
    assert len(x) > 0, "shape must have at least one dimension"
    vals = tf.strings.join([
        tf.strings.as_string(x_i) if x_i is not None else tf.constant("?")
        for x_i in x
    ], separator=" ")
    return tf.strings.join([
        "[",
        vals,
        "]",
    ])

def tf_dict_repr(x, indent=_default_indent, depth=0, prefix=""):
    return tf.strings.join([
        indent*depth,
        prefix,
        "{\n",
        tf.strings.join([
            tf.strings.join([
                tf_repr(v, indent=indent, depth=depth+1, prefix=tf.strings.join([k, ": "])),
                ",\n",
            ])
            for k, v in x.items()
        ]),
        indent*depth,
        "}",
    ])

def tf_tuple_repr(x, indent=_default_indent, depth=0, prefix=""):
    return tf.strings.join([
        indent*depth,
        prefix,
        "(\n",
        tf.strings.join([
            tf.strings.join([
                tf_repr(v, indent=indent, depth=depth+1),
                ",\n",
            ])
            for v in x
        ]),
        indent*depth,
        ")",
    ])

def tf_list_repr(x, indent=_default_indent, depth=0, prefix=""):
    return tf.strings.join([
        indent*depth,
        prefix,
        "[\n",
        tf.strings.join([
            tf_repr(x[0], indent=indent, depth=depth+1),
            ",\n",
            indent*(depth+1),
            "...\n",
        ]),
        indent*depth,
        "]",
    ])

SomeT = TypeVar("SomeT")
@export
def tf_print(s: SomeT, tag: str = None, output_stream=sys.stdout) -> SomeT:
    """
    If debug mode, then pretty-print tensorflow tensors, including within graph mode.
    Returns the input so it can be used in an expression.
    """
    if tag is None:
        tag = ""
        tail = ""
        depth = 0
    else:
        tag = "--- @ " + tag + "\n"
        tail = "\n---"
        depth = 1
    if tf.is_tensor(s) and is_keras_tensor(s):
        s = s.type_spec
    val = tf.strings.join([tag, tf_repr(s, depth=depth), tail])
    if tf.executing_eagerly():
        print(val.numpy().decode('utf-8'), flush=True)
    else:
        tf.print(val, output_stream=output_stream)

@export
def dbg(s: SomeT, tag: str = None) -> SomeT:
    """
    If debug mode, then pretty-print tensorflow tensors, including within graph mode.
    Returns the input so it can be used in an expression.
    """
    if debug:
        tf_print(s, tag=tag, output_stream=sys.stderr)
    return s


@export
def tf_str(x, indent=_default_indent, depth=0, prefix="") -> str:
    return tf_repr(x, indent=indent, depth=depth, prefix=prefix).numpy().decode("utf-8")

def docstring_for_tf_repr_and_friends(fn_name):
    return """
Implements a human-friendly of """ + fn_name + """ for tensorflow objects,
which can be used in a tf.function.

>>> tf_print(tf.constant(1))
1
>>> tf_str(tf.constant(1))
'1'
>>> tf_repr(tf.constant(1))
<tf.Tensor: shape=(), dtype=string, numpy=b'1'>
>>> tf_print(tf.zeros([3]))
●float32[3]
>>> tf_str(tf.zeros([3]))
'●float32[3]'
>>> tf_print(tf.TensorSpec([2, 3], tf.int32))
○int32[2 3]
>>> tf_str(tf.TensorSpec([2, 3], tf.int32))
'○int32[2 3]'
>>> tf_print(tf.zeros([2, 3]))
●float32[2 3]
>>> tf_print({ "a": tf.zeros([2, 3]), "b": tf.zeros([3]) })
{
  a: ●float32[2 3],
  b: ●float32[3],
}
>>> tf_print([ tf.zeros([2, 3]), tf.zeros([2, 3]) ])
[
  ●float32[2 3],
  ...
]
>>> tf_print(( tf.zeros([2, 3]), tf.zeros([2, 3, 5]) ))
(
  ●float32[2 3],
  ●float32[2 3 5],
)
>>> tf_print(tf.data.Dataset.range(10))
●Dataset[10] ○int64[]
>>> tf_print(tft.DatasetSpec(tf.TensorSpec([], tf.int64)))
○Dataset[?] ○int64[]
"""

tf_repr.__doc__ = docstring_for_tf_repr_and_friends("repr()")
tf_str.__doc__ = docstring_for_tf_repr_and_friends("str()")
tf_print.__doc__ = docstring_for_tf_repr_and_friends("print()")



@export
def multidim_idxs_to_flat_idxs(idxs, shape):
    """
    Convert a list of multidimensional indices to a list of flat indices.
    Supports any number of batch dimensions.

    >>> list(multidim_idxs_to_flat_idxs(
    ...     idxs=[[0, 0],
    ...           [0, 1],
    ...           [1, 0],
    ...           [1, 1]],
    ...     shape=[2, 2]
    ... ).numpy())
    [0, 1, 2, 3]
    >>> list(multidim_idxs_to_flat_idxs(
    ...     idxs=[[0],
    ...           [1],
    ...           [2]],
    ...     shape=[5],
    ... ).numpy())
    [0, 1, 2]
    >>> tf_print(multidim_idxs_to_flat_idxs(
    ...     idxs=[[[0],
    ...            [1],
    ...            [2]]],
    ...     shape=[5],
    ... ))
    ●int32[1 3]
    >>> tf_print(multidim_idxs_to_flat_idxs(
    ...     idxs=[[[[0],
    ...             [1],
    ...             [2]]]],
    ...     shape=[5],
    ... ))
    ●int32[1 1 3]
    """

    idxs = tf.convert_to_tensor(idxs)
    shape = tf.convert_to_tensor(shape)

    assert idxs.shape[-1] == len(shape)

    if len(shape) == 1:
        return idxs[..., 0]

    # Multiply the indices by the strides, and sum them.
    # compute strides
    strides = tf.math.cumprod(shape, exclusive=True, reverse=True)
    # broadcast strides have the same number of batch dimensions as idxs
    strides = tf.reshape(strides, [1] * (len(idxs.shape) - 1) + [-1])
    # broadcast idxs to the same shape as strides
    idxs = idxs + tf.zeros_like(strides)
    # multiply and sum
    return tf.reduce_sum(idxs * strides, axis=-1)


@export
def inspect(fn, tag: str|None = None):
    @tf.function
    def inspect_fn(*args, **kwargs):
        if len(args) == 1:
            val_fmt = "{}"
            val = tf_repr(args[0])
        else:
            val_fmt = "args: {}"
            val = tf_tuple_repr(args)

        if len(kwargs) == 0:
            kwarg_fmt = ""
            vals = [val]
        else:
            kward_fmt = ", kwargs: {}"
            vals = [val, tf_dict_repr(kwargs)]

        format = f"inspect: function '{fn.__name__}' called with" + val_fmt + kwarg_fmt
        if tag is not None:
            format = f"#{tag} " + format

        tf.print(tf.strings.format(format, vals))

        return fn(*args, **kwargs)
    return inspect_fn

@export
def count_calls(fn) -> tuple[Callable, tf.Variable]:
    count = tf.Variable(0, dtype=tf.int32, synchronization=tf.VariableSynchronization.ON_READ, aggregation=tf.VariableAggregation.SUM)

    def count_calls_fn(*args, **kwargs):
        count.assign_add(1)
        return fn(*args, **kwargs)

    return count_calls_fn, count


def _fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def _stream_redirected(stream, to):

    stdout_fd = _fileno(stream)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stream.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(_fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stream # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stream.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

@export
@contextmanager
def stderr_captured(to=os.devnull):
    with _stream_redirected(sys.stderr, to) as stderr:
        yield stderr

@export
@contextmanager
def stdout_captured(to=os.devnull):
    with _stream_redirected(sys.stdout, to) as stdout:
        yield stdout


def ensure_suffix(path: os.PathLike, suffix: str) -> Path:
    suffix = suffix.lstrip(".")
    path = Path(path)
    if path.suffix != f".{suffix}":
        path = path.with_suffix(f".{suffix}")
    return path


if __name__ == '__main__':
    import doctest
    doctest.testmod()
