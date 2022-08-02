"""

Code lifted from HuggingFace Tensorflow Deberta implementation:
https://github.com/huggingface/transformers/blob/d0acc9537829e7d067edbb791473bbceb2ecf056/src/transformers/models/deberta/modeling_tf_deberta.py
Under the following licence:

Copyright 2021 Microsoft and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers, Model, Input
import numpy as np
from typing import Optional, Tuple, Union, List

def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
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

def get_mask(input, dropout):
    mask = tf.cast(
        1 - tf.compat.v1.distributions.Bernoulli(probs=1 - dropout).sample(sample_shape=shape_list(input)), tf.bool
    )
    return mask, dropout

@tf.custom_gradient
def TFDebertaXDropout(input, local_ctx):
    mask, dropout = get_mask(input, local_ctx)
    scale = tf.convert_to_tensor(1.0 / (1 - dropout), dtype=tf.float32)
    input = tf.cond(dropout > 0, lambda: tf.where(mask, 0.0, input) * scale, lambda: input)

    def custom_grad(upstream_grad):
        return tf.cond(
            scale > 1, lambda: (tf.where(mask, 0.0, upstream_grad) * scale, None), lambda: (upstream_grad, None)
        )

    return input, custom_grad

class TFDebertaStableDropout(tf.keras.layers.Layer):
    """
    Optimized dropout module for stabilizing the training
    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = tf.convert_to_tensor(drop_prob, dtype=tf.float32)

    def call(self, inputs: tf.Tensor, training: tf.Tensor = False):
        if training and self.drop_prob > 0:
            return TFDebertaXDropout(inputs, self.drop_prob)
        return inputs

def stable_softmax(logits: tf.Tensor, axis: Optional[int] = None, name: Optional[str] = None) -> tf.Tensor:
    """
    Stable wrapper that returns the same output as `tf.nn.softmax`, but that works reliably with XLA on CPU. It is
    meant as a workaround for the [following issue](https://github.com/tensorflow/tensorflow/issues/55682), and will be
    removed after it gets fixed. The arguments and outputs are the same as `tf.nn.softmax`, and relies on the fact that
    `softmax(x) = softmax(x + c)` (see https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html).
    Args:
        logits (`tf.Tensor`):
            Must be one of the following types: half, float32, float64.
        axis (`int`, *optional*):
            The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
        name (`str`, *optional*):
            A name for the operation.
    Returns:
        `tf.Tensor`:
            A Tensor. Has the same type and shape as logits.
    """
    # TODO: When the issue linked above gets sorted, add a check on TF version here and use the original function if
    # it has the fix. After we drop the support for unfixed versions, remove this function.
    return tf.nn.softmax(logits=logits + 1e-9, axis=axis, name=name)

class TFDebertaXSoftmax(tf.keras.layers.Layer):
    """
    Masked Softmax which is optimized for saving memory
    Args:
        input (`tf.Tensor`): The input tensor that will apply softmax.
        mask (`tf.Tensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor, mask: tf.Tensor):

        rmask = tf.logical_not(tf.cast(mask, tf.bool))
        output = tf.where(rmask, float("-inf"), inputs)
        output = stable_softmax(output, self.axis)
        output = tf.where(rmask, 0.0, output)
        return output

def get_initializer(initializer_range: float = 0.02) -> tf.initializers.TruncatedNormal:
    """
    Creates a `tf.initializers.TruncatedNormal` with the given range.
    Args:
        initializer_range (*float*, defaults to 0.02): Standard deviation of the initializer range.
    Returns:
        `tf.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

ACT2FN = {
    # "gelu_10": gelu_10,
    # "gelu_fast": gelu_fast,
    # "gelu_new": gelu_new,
    # "glu": glu,
    # "mish": mish,
    # "quick_gelu": quick_gelu,
    "gelu": tf.keras.activations.gelu,
    "relu": tf.keras.activations.relu,
    "sigmoid": tf.keras.activations.sigmoid,
    "silu": tf.keras.activations.swish,
    "swish": tf.keras.activations.swish,
    "tanh": tf.keras.activations.tanh,
}

def get_tf_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")

class TFDebertaIntermediate(layers.Layer):
    def __init__(self, intermediate_size, initializer_range, hidden_act, **kwargs):
        super().__init__(**kwargs)

        self.dense = layers.Dense(
            units=intermediate_size, kernel_initializer=get_initializer(initializer_range), name="dense"
        )

        if isinstance(hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(hidden_act)
        else:
            self.intermediate_act_fn = hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

class TFDebertaOutput(layers.Layer):
    def __init__(self, initializer_range, hidden_size, hidden_dropout_prob, layer_norm_eps, **kwargs):
        super().__init__(**kwargs)

        self.dense = layers.Dense(
            units=hidden_size, kernel_initializer=get_initializer(initializer_range), name="dense"
        )
        self.LayerNorm = layers.LayerNormalization(epsilon=layer_norm_eps, name="LayerNorm")
        self.dropout = TFDebertaStableDropout(hidden_dropout_prob, name="dropout")

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
