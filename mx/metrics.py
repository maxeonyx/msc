
import abc

from mx.tf import *
from mx.utils import tf_scope


class MxMetric(abc.ABC, tf.Module):

    def __init__(self, name: str, unit: str | None, reset_every_epoch: bool):
        super().__init__(name=name)
        self.reset_every_epoch = reset_every_epoch
        self.initialized = tf.Variable(False, trainable=False, name="initialized")
        self.unit = unit

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, **inputs: TensorLike) -> tf.Tensor:
        pass

    @abc.abstractmethod
    def result(self) -> np.float32 | np.float64 | np.int32 | np.ndarray | None:
        pass

class TimeSinceLastCall(MxMetric):
    def __init__(self, name="time_since_last_call", reset_every_epoch=True):
        super().__init__(
            name=name,
            unit="s",
            reset_every_epoch=reset_every_epoch
        )
        self.last_call = tf.Variable(tf.timestamp(), dtype=tf.float64, trainable=False, name="last_call")

    def reset(self):
        self.last_call.assign(tf.timestamp())
        self.initialized.assign(False)

    @tf_scope
    def update(self, inputs):
        self.initialized.assign(True)
        timestamp = tf.timestamp()
        result = timestamp - self.last_call
        self.last_call.assign(timestamp)
        return result

    @tf_scope
    def result(self):
        if not self.initialized.numpy():
            return None
        return (tf.timestamp() - self.last_call).numpy()

class RunningMean(MxMetric):
    
    @tf_scope
    def __init__(self, fn, unit: str = None, element_shape=[], dtype=tf.float64, name="running_mean", reset_every_epoch=True):
        if isinstance(fn, MxMetric):
            unit = fn.unit
            self.fn = fn.update
        else:
            self.fn = fn
        super().__init__(name=name, unit=unit, reset_every_epoch=reset_every_epoch)
        self.total = tf.Variable(initial_value=tf.zeros(element_shape, dtype=dtype), name="total", trainable=False)
        self.count = tf.Variable(0, dtype=tf.int64, name="count", trainable=False)
    
    def reset(self):
        self.total.assign(tf.zeros_like(self.total))
        self.count.assign(0)
        self.initialized.assign(False)

    @tf_scope
    def update(self, inputs):
        self.initialized.assign(True)
        val = self.fn(inputs)
        self.total.assign_add(tf.cast(val, self.total.dtype))
        self.count.assign_add(1)
        return self._result()

    @tf_scope
    def _result(self):
        return self.total / tf.cast(self.count, self.total.dtype)

    def result(self):
        if not self.initialized.numpy():
            return None
        return self._result().numpy()

class Rolling(MxMetric):

    @tf_scope
    def __init__(self, length, fn, unit: str, element_shape=[], dtype=tf.float32, reduction_fn=tf.reduce_mean, name="rolling", reset_every_epoch=True):
        if isinstance(fn, MxMetric):
            unit = fn.unit
            self.fn = fn.update
        else:
            self.fn = fn
        super().__init__(name=name, unit=unit, reset_every_epoch=reset_every_epoch)
        self.length = tf.constant(length, tf.int64)
        self.reduction_fn = reduction_fn
        self.buffer = tf.Variable(
            initial_value=tf.zeros(shape=[length] + element_shape, dtype=dtype),
            name="history",
            trainable=False,
            aggregation=tf.VariableAggregation.SUM,
            synchronization=tf.VariableSynchronization.ON_READ,
        )
        self.index = tf.Variable(
            initial_value=tf.constant(0, dtype=tf.int64),
            name="index",
            trainable=False,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            synchronization=tf.VariableSynchronization.ON_READ,
        )
        
    def reset(self):
        self.index.assign(0)
        self.initialized.assign(False)

    @tf_scope
    def update(self, inputs):
        self.initialized.assign(True)
        self.index.assign_add(1)
        i = self.index % self.length
        val = self.fn(inputs)
        self.buffer[i].assign(val)
        return self._result()
    
    @tf_scope
    def _result(self):
        i = tf.math.minimum(self.index, self.length)
        return self.reduction_fn(self.buffer[:i])
    
    def result(self):
        if not self.initialized.numpy():
            return None
        return self._result().numpy()

class InstantaneousMetric(MxMetric):
    def __init__(self, fn, unit: str, name="instantaneous", reset_every_epoch=True):
        if isinstance(fn, MxMetric):
            unit = fn.unit
            self.fn = fn.update
        else:
            self.fn = fn
        super().__init__(name=name, unit=unit, reset_every_epoch=reset_every_epoch)
        self.val = tf.Variable(0., trainable=False, name="val")
    
    def reset(self):
        self.initialized.assign(False)
    @tf_scope
    def update(self, inputs):
        self.initialized.assign(True)
        result = self.fn(inputs)
        self.val.assign(result)
        return result

    @tf_scope
    def result(self):
        if not self.initialized.numpy():
            return None
        return self.val.numpy()

def wrap_loss_fn_for_metrics(loss_fn):
    def wrapped_loss_fn(inputs):
        return loss_fn(inputs["targets"], inputs["outputs"])
    return wrapped_loss_fn
