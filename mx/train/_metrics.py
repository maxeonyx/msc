
import abc

from mx.utils.tf import *
from mx.utils import tf_scope

class MxMetric(abc.ABC, tf.Module):

    def __init__(self, name: str, reset_every_epoch: bool = True):
        super().__init__(name=name)
        self.reset_every_epoch = reset_every_epoch
        self.initialized = False

    @abc.abstractmethod
    def reset(self):
        pass

    @property
    @abc.abstractmethod
    def result(self) -> TensorLike:
        pass

    @abc.abstractmethod
    def update(self, **inputs: TensorLike):
        pass
    
    @property
    @abc.abstractmethod
    def unit(self) -> str:
        pass

class TimeSinceLastCall(MxMetric):
    def __init__(self, name="time_since_last_call", **kwargs):
        super().__init__(name=name, **kwargs)
        self.last_call = tf.Variable(0., dtype=tf.float64, trainable=False, name="last_call")

    def reset(self):
        self.last_call.assign(tf.timestamp())
        self.initialized = False
    
    def unit(self) -> str:
        return "s"

    @property
    @tf_scope
    def result(self):
        return tf.timestamp() - self.last_call

    @tf_scope
    def update(self, inputs):
        self.initialized = True
        timestamp = tf.timestamp()
        result = self.result
        self.last_call.assign(timestamp)
        return result

class RunningMean(MxMetric):
    
    @tf_scope
    def __init__(self, fn, unit: str = None, element_shape=[], dtype=tf.float32, name="running_mean", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = tf.Variable(initial_value=tf.zeros(element_shape, dtype=dtype), name="total", trainable=False)
        self.count = tf.Variable(0, dtype=tf.int64, name="count", trainable=False)
        if isinstance(fn, MxMetric):
            self._unit = fn.unit
            self.fn = fn.update
        else:
            self._unit = unit
            self.fn = fn

    @property
    def unit(self) -> str:
        return self._unit
    
    def reset(self):
        self.total.assign(tf.zeros_like(self.total))
        self.count.assign(0)
        self.initialized = False

    @property
    @tf_scope
    def result(self):
        return self.total / tf.cast(self.count, self.total.dtype)

    @tf_scope
    def update(self, inputs):
        self.initialized = True
        val = self.fn(inputs)
        self.total.assign_add(val)
        self.count.assign_add(1)
        return self.result

class Rolling(MxMetric):

    @tf_scope
    def __init__(self, length, fn, unit: str, element_shape=[], dtype=tf.float32, reduction_fn=tf.reduce_mean, name="rolling", **kwargs):
        super().__init__(name=name, **kwargs)
        self.length = tf.constant(length, tf.int64)
        self.reduction_fn = reduction_fn
        if isinstance(fn, MxMetric):
            self._unit = fn.unit
            self.fn = fn.update
        else:
            self._unit = unit
            self.fn = fn
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

    @property
    def unit(self) -> str:
        return self._unit
    
    def reset(self):
        self.index.assign(0)
        self.buffer.assign(tf.zeros_like(self.buffer))
        self.initialized = False
    
    @property
    @tf_scope
    def result(self):
        i = tf.math.minimum(self.index, self.length)
        return self.reduction_fn(self.buffer[:i])

    @tf_scope
    def update(self, inputs):
        self.initialized = True
        self.index.assign_add(1)
        i = self.index % self.length
        val = self.fn(inputs)
        self.buffer[i].assign(val)
        return self.result

class InstantaneousMetric(MxMetric):
    def __init__(self, fn, unit: str, name="instantaneous", **kwargs):
        super().__init__(name=name, **kwargs)
        if isinstance(fn, MxMetric):
            self._unit = fn.unit
            self.fn = fn.update
        else:
            self._unit = unit
            self.fn = fn
        self.val = tf.Variable(0., trainable=False, name="val")

    @property
    def unit(self) -> str:
        return self._unit
    
    def reset(self):
        self.val.assign(0.)
        self.initialized = False

    @tf_scope
    def update(self, inputs):
        self.initialized = True
        result = self.fn(inputs)
        self.val.assign(result)
        return result

    @property
    @tf_scope
    def result(self):
        return self.val
