from __future__ import annotations

import typing

from tensorflow.python.types.core import TensorLike, Value as ValueTensor
from tensorflow.python.framework.dtypes import DType
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.data.util.structure import NoneTensorSpec, NoneTensor
from tensorflow import TensorSpec

NestedTensor = typing.Union[
    Tensor,
    dict[str, 'NestedTensor'],
    list['NestedTensor'],
    tuple['NestedTensor'],
]
