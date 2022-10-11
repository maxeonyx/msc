import typing
if typing.TYPE_CHECKING:
    from tensorflow.python.types.core import TensorLike, Value as ValueTensor
    from tensorflow.python.framework.dtypes import DType
    from tensorflow.python.framework.tensor_shape import TensorShape
    from tensorflow.python.framework.ops import Tensor
    from tensorflow import TensorSpec
