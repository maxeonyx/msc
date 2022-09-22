
import abc
from dataclasses import dataclass
from typing import Callable, Generic, Literal, Optional, Set, TypeVar, Union
from typing_extensions import LiteralString

import tensorflow as tf
import typing
if typing.TYPE_CHECKING:
    import keras.api._v2.keras as keras
    from keras.api._v2.keras import Model, Input, layers
    from tensorflow.python.types.core import TensorLike
else:
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers
    from tensorflow.types.experimental import TensorLike

from mx.utils import tf_scope

StepMetric = Literal[
    "loss",
    "step_time",
]

Types = TypeVar("Types", )
@dataclass(frozen=True)
class MetricCfg(Generic[Types]):
    type: Types
    name: Optional[str] = None
    reset_every_epoch: bool = True
    """If True, the metric will reset every epoch."""

@dataclass(frozen=True)
class RollingAvgMetricCfg(MetricCfg[StepMetric]):
    """Rolling-average loss metric. Maintains an average of the last `n_steps` steps."""

    n_steps: int = 100
    """Number of steps to average over"""

@dataclass(frozen=True)
class RunningAvgMetricCfg(MetricCfg[StepMetric]):
    """
    Running-average loss metric. Maintains an average of all steps since the
    last reset.
    """

@dataclass(frozen=True)
class InstantaneousMetricCfg(MetricCfg[StepMetric | Literal["total_epoch_time"]]):
    """Instantaneous loss metric. Simply returns the most-recently computed value."""



class MyMetric(abc.ABC, layers.Layer, Callable):

    def __init__(self, name: str):
        super().__init__(name=name)

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def result(self) -> TensorLike:
        pass

    @abc.abstractmethod
    def __call__(self, i_step: int, inputs: dict[str, TensorLike]):
        pass

class RunningMean(MyMetric):
    
    @tf_scope
    def __init__(self, fn, element_shape=[], dtype=tf.float32, name="running_mean"):
        super().__init__(name=name)
        self.total = tf.Variable(initial_value=tf.zeros(element_shape, dtype=dtype), name="total", trainable=False)
        self.count = tf.Variable(0., dtype=tf.float32, name="count", trainable=False)
        self.fn = fn
    
    def reset(self):
        self.total.assign(tf.zeros_like(self.total))
        self.count.assign(0.)

    @tf_scope
    def result(self):
        return self.total / self.count

    @tf_scope
    def __call__(self, inputs):
        val = self.fn(inputs)
        self.total.assign_add(val),
        self.count.assign_add(1.)

class TimeSinceLastCall(MyMetric):
    def __init__(self, name="time_since_last_call"):
        super().__init__(name=name)
        self.last_call = tf.Variable(0., dtype=tf.float64, trainable=False, name="last_call")

    def reset(self):
        self.last_call.assign(tf.timestamp())

    def result(self):
        return tf.timestamp() - self.last_call

    def __call__(self, inputs):
        timestamp = tf.timestamp()
        since_last_call = timestamp - self.last_call
        self.last_call.assign(timestamp)
        return since_last_call

class Rolling(MyMetric):

    @tf_scope
    def __init__(self, length, fn, element_shape=[], dtype=tf.float32, reduction_fn=tf.reduce_mean, name="rolling"):
        super().__init__(name=name)
        self.length = length
        self.reduction_fn = reduction_fn
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
    
    def reset(self):
        self.index.assign(0)
        self.buffer.assign(tf.zeros_like(self.buffer))

    @tf_scope
    def __call__(self, inputs):
        self.index.assign_add(1)
        i = self.index % self.length
        val = self.fn(inputs)
        self.buffer[i].assign(val)
        return self.result()
    
    @tf_scope
    def result(self):
        i = tf.math.minimum(self.index, self.length)
        return self.reduction_fn(self.buffer[:i])

def make_metric(cfg: MetricCfg, loss_fn: Optional[Callable]) -> MyMetric:

    if cfg.type == "loss":
        if loss_fn is None:
            raise ValueError("Must provide a loss function for loss metrics")
        fn = loss_fn
    elif cfg.type == "step_time":
        fn = TimeSinceLastCall()
    elif cfg.type == "total_epoch_time":
        fn = TimeSinceLastCall()
    else:
        raise ValueError(f"Unknown metric type {cfg.type}")

    if isinstance(cfg, RollingAvgMetricCfg):
        autoname = f"{cfg.type}_avg_{cfg.n_steps}"
        return Rolling(
            length=cfg.steps,
            fn=fn,
            name=cfg.name or autoname,
        )
    elif isinstance(cfg, RunningAvgMetricCfg):
        autoname=f"{cfg.type}_epoch_avg",
        return RunningMean(
            fn=fn,
            name=cfg.name or autoname,
        )
    elif isinstance(cfg, InstantaneousMetricCfg):
        autoname=f"{cfg.type}_instant",
        return RunningMean(
            fn=fn,
            name=cfg.name or autoname,
        )
    else:
        raise ValueError(f"Unknown metric config: {cfg}")
