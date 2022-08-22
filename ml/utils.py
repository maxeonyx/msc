from ast import arg
import functools
import inspect
from operator import is_
from tensorflow import keras
import tensorflow as tf
from math import pi, tau

def multidim_indices(shape, flatten=True):
    """
    Uses tf.meshgrid to get the indices for a tensor of any rank
    Returns an int32 tensor of shape [ product(shape), rank ]
    """
    indices = tf.meshgrid(*[tf.range(s) for s in shape])
    indices = tf.stack(indices, axis=-1)
    if flatten:
        indices = tf.reshape(indices, [-1, len(shape)])
    return indices


def multidim_indices_of(tensor, flatten=True):
    """
    Uses tf.meshgrid to get the indices for a tensor of any rank
    Returns an int32 tensor of shape [ product(shape), rank ]
    """
    shape = tf.shape(tensor)
    return multidim_indices(shape, flatten=flatten)


def angle_wrap(angles):
    """
    Wrap angle in radians to [-pi, pi] range
    """
    angles = (angles + pi) % tau - pi
    # angles = tf.math.atan2(tf.sin(angles), tf.cos(angles))
    return angles


def circular_mean(angles, axis=0):
    # compute the circular mean of the data for this example+track
    # rotate the data so that the circular mean is 0
    # store the circular mean
    means_cos_a = tf.reduce_mean(tf.math.cos(angles), axis=axis)
    means_sin_a = tf.reduce_mean(tf.math.sin(angles), axis=axis)
    circular_means = tf.math.atan2(means_sin_a, means_cos_a)
    return circular_means


def recluster(angles, circular_means=None, frame_axis=0):
    if circular_means is None:
        circular_means = circular_mean(angles, axis=frame_axis)

    # rotate the data so the circular mean is 0
    angles = angles - tf.expand_dims(circular_means, axis=frame_axis)
    angles = angle_wrap(angles)

    return angles


def unrecluster(angles, circular_means, frame_axis=0):
    # assuming the mean is currently 0, rotate the data so the mean is
    # back to the original given by `circular_means`
    angles = angles + tf.expand_dims(circular_means, axis=frame_axis)
    angles = angle_wrap(angles)

    return angles


class WarmupLRSchedule(keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, peak_learning_rate, warmup_steps, other_schedule=None):
        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps
        self.other_schedule = other_schedule
        if other_schedule is None:
            self.other_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=peak_learning_rate,
                decay_steps=1000,
                decay_rate=0.96
            )

    def __call__(self, step):
        return tf.cond(
            pred=tf.less(step, self.warmup_steps),
            true_fn=lambda: self.peak_learning_rate *
            tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32),
            false_fn=lambda: self.other_schedule(step)
        )

    def get_config(self):
        return {
            'peak_learning_rate': self.peak_learning_rate,
            'warmup_steps': self.warmup_steps,
            'other_schedule': self.other_schedule
        }


class KerasLossWrapper(tf.keras.metrics.Metric):
    """
    Show instantaneous loss in keras progress bar.
    """

    def __init__(self, loss_fn, name="instantaneous_loss", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.loss_fn = loss_fn
        self.total = self.add_weight(name="total", initializer="zeros")

    def reset_state(self):
        pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = self.loss_fn(y_true, y_pred)
        self.total.assign(loss)

    def result(self):
        return self.total

from tensorflow.python.util import tf_decorator
from tensorflow.python.module.module import camel_to_snake
import inspect


def tf_scope(func):
    """Decorator to automatically enter the module name scope.
    >>> class MyModule(tf.Module):
    ...   @tf.Module.with_name_scope
    ...   def __call__(self, x):
    ...     if not hasattr(self, 'w'):
    ...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
    ...     return tf.matmul(x, self.w)
    Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
    names included the module name:
    >>> mod = MyModule()
    >>> mod(tf.ones([1, 2]))
    <tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
    >>> mod.w
    <tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
    numpy=..., dtype=float32)>
    Args:
        method: The method to wrap.
    Returns:
        The original method wrapped such that it enters the module's name scope.
    """

    is_init = func.__name__ == "__init__"
    is_call = func.__name__ == "__call__"
    
    fn_name = func.__name__

    @functools.wraps(func)
    def func_with_name_scope(*args, **kwargs):
        is_module = len(args) > 0 and isinstance(args[0], tf.Module)
        
        if is_module and is_init:
            # init happens before the tf.Module instance has a _name attribute
            if 'name' in kwargs and kwargs['name'] is not None:
                name_prefix = kwargs['name']
            else:
                name_prefix = camel_to_snake(type(args[0]).__name__)
            
            scope_name = name_prefix + "_init"
        elif is_module and not is_call:
            scope_name = args[0].name + "_" + fn_name
        elif is_module and is_call:
            scope_name = args[0].name
        else:
            scope_name = fn_name
        
        with tf.name_scope(scope_name):
            return func(*args, **kwargs)
    
    return tf_decorator.make_decorator(func, func_with_name_scope)
