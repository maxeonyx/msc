from typing import List, Tuple, Union

import tensorflow as tf
import numpy as np

Shape = Union[List[int], tf.TensorShape]

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

def input_dict(*arr):
    return {
        inp.name: inp for inp in arr
    }

def make_causal_mask(n) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    mask = tf.linalg.band_part(tf.ones([1, n, n]), -1, 0)
    scales = 1./tf.sqrt(tf.reduce_sum(mask, axis=-1))
    return mask, scales
