from mx.prelude import *

@export
class MxLayer(tf.Module, Callable):
    pass

@export
def make_causal_mask(m: int, n: int = None) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.

    If n is None, return a square mask of shape (1, m, m)
    Otherwise, returns a mask with shape (1, m, n)

    e.g. if m = 3, n = 4
    [[[1, 0, 0, 0],
      [1, 1, 0, 0],
      [1, 1, 1, 0]]]

    e.g. if m = n = 3
    [[[1, 0, 0],
      [1, 1, 0],
      [1, 1, 1]]]

    e.g. if m = 4, n = 3
    [[[1, 0, 0],
      [1, 1, 0],
      [1, 1, 1],
      [1, 1, 1]]]
    """
    if n is None:
        n = m

    mask = tf.linalg.band_part(tf.ones([1, m, n]), -1, 0)
    scales = 1./tf.sqrt(tf.reduce_sum(mask, axis=-1))
    return mask, scales
