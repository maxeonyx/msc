from __future__ import annotations as _annotations

import typing as _typing

from tensorflow.python.types.core import TensorLike, Value as ValueTensor
from tensorflow.python.framework.dtypes import DType
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.data.util.structure import NoneTensorSpec, NoneTensor
from tensorflow import TensorSpec

if _typing.TYPE_CHECKING:
    from tensorflow.python.types.core import GenericFunction
    from tensorflow.python.data.ops.dataset_ops import DatasetSpec
else:
    from tensorflow.types.experimental import GenericFunction
    from tensorflow.data import DatasetSpec

NestedTensor = _typing.Union[
    Tensor,
    dict[str, 'NestedTensor'],
    list['NestedTensor'],
    tuple['NestedTensor'],
]
