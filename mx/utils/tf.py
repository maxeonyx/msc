from math import pi, tau

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda/"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

import numpy as np
import einops as ein
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import typing
if typing.TYPE_CHECKING:
    import keras.api._v2.keras as keras
    from keras.api._v2.keras import Model, Input, layers
    from keras.api._v2.keras.backend import is_keras_tensor
    from tensorflow.python.types.core import TensorLike
    import tensorflow.python.types.core as tft
else:
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers
    from tensorflow.keras.backend import is_keras_tensor
    from tensorflow.types.experimental import TensorLike
    import tensorflow.types.experimental as tft



def shape_list(tensor: typing.Union[tf.Tensor, np.ndarray]) -> list[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.
    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.
    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

@tf.function
def tf_val_repr(x, indent="    ", depth=0, prefix=""):
    if isinstance(x, dict):
        return tf_dict_repr(x, indent=indent, depth=depth, prefix=prefix)
    elif isinstance(x, tuple):
        return tf_tuple_repr(x, indent=indent, depth=depth, prefix=prefix)
    elif len(x.shape) == 0:
        if x.dtype == tf.string:
            str_x = tf.strings.join(["\"", x, "\""], separator="")
        else:
            str_x = tf.strings.as_string(x)
    else:
        str_x = tf.strings.join([
            "`",
            tf.constant(x.dtype.name, tf.string),
            "[",
            tf.strings.reduce_join(tf.strings.as_string(tf.shape(x)), separator=" "),
            "]",
        ])
    
    return tf.strings.join([
        *([indent]*depth),
        prefix,
        str_x,
    ])

@tf.function
def tf_dict_repr(x, indent="    ", depth=0, prefix=""):
    return tf.strings.join([
        *([indent]*depth),
        prefix,
        "{\n",
        tf.strings.join([
            tf.strings.join([
                tf_val_repr(v, indent=indent, depth=depth+1, prefix=tf.strings.join([k, ": "], separator="")),
                ",\n",
            ], separator="")
            for k, v in x.items()
        ], separator=""),
        *([indent]*depth),
        "}",
    ], separator="")

@tf.function
def tf_tuple_repr(x, indent="    ", depth=0, prefix=""):
    return tf.strings.join([
        *([indent]*depth),
        prefix,
        "(\n",
        tf.strings.join([
            tf.strings.join([
                tf_val_repr(v, indent=indent, depth=depth+1),
                ",\n",
            ], separator="")
            for v in x
        ], separator=""),
        *([indent]*depth),
        ")",
    ], separator="")

def inspect(tag):
    @tf.function
    def inspect_fn(*args):
        if len(args) == 1:
            val = tf_val_repr(args[0])
        else:
            val = tf_tuple_repr(args)
            
        tf.print(tf.strings.join([
            "inspect@",
            tag,
            ":\n",
            val,
            "\n",
        ], separator=""))
        
        if len(args) == 1:
            return args[0]
        else:
            return args
    return inspect_fn

def count_calls():
    count = tf.Variable(0, dtype=tf.int32, synchronization=tf.VariableSynchronization.ON_READ, aggregation=tf.VariableAggregation.SUM)
    @tf.function
    def count_calls_fn(*args):
        count.assign_add(1)
        return args
    return count_calls_fn, count
