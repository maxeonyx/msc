from sklearn.cluster import MiniBatchKMeans
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model, Input, layers
from IPython.display import display
import tensorflow_datasets as tfds
import time
import matplotlib.pyplot as plt
import enlighten
import tensorflow_probability as tfp
from dotmap import DotMap

# class WarmupInvSquare(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, embd_dim, warmup_steps):
#         super(WarmupInvSquare, self).__init__()

#         self.d_model = embd_dim
#         self.d_model = tf.cast(self.d_model, tf.float32)

#         self.warmup_steps = warmup_steps

#     def __call__(self, step):
#         arg1 = step * (self.warmup_steps ** -1.5)
#         arg2 = tf.math.rsqrt(step)

#         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class Linear(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, start_lr):
        super(Linear, self).__init__()
        
        self.start_lr = tf.cast(start_lr, tf.float64)
        self.total_steps = tf.cast(total_steps, tf.float64)

    def __call__(self, step):
        step = tf.cast(step, tf.float64)
        decay = 1 - step/self.total_steps
        
        return self.start_lr * decay

class Constant(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, rate):
        super(Constant, self).__init__()
        self.rate = tf.cast(rate, tf.float64)
    
    def __call__(self, step):
        return tf.ones_like(step, dtype=tf.float64)*self.rate

class Warmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, other_sched, warmup_steps, end_lr):
        super(Warmup, self).__init__()
        self.end_lr = tf.cast(end_lr, tf.float64)
        self.other_sched = other_sched
        self.warmup_steps = tf.cast(warmup_steps, tf.float64)
    
    def __call__(self, step):
        float_step = tf.cast(step, tf.float64)
        arg1 = tf.math.minimum(self.end_lr * float_step/self.warmup_steps, self.end_lr)
        arg2 = self.other_sched(step)
        return tf.math.minimum(arg1, arg2)


class Exponential(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, peak_lr, final_lr):
        super(Exponential, self).__init__()
        self.scale = tf.cast(peak_lr, tf.float64)
        self.rate = tf.cast(total_steps/tf.math.log(peak_lr/final_lr), tf.float64)
    
    def __call__(self, step):
        step = tf.cast(step, tf.float64)
        arg1 = tf.math.exp(-step / self.rate)
        return self.scale * arg1
    
def exponential_batch_size(total_steps, initial, final):
    
    scale = tf.cast(initial, tf.float64)
    rate = tf.cast(total_steps/tf.math.log(initial/final), tf.float64)
    
    def call(step):
        step = tf.cast(step, tf.float64)
        return tf.cast(tf.math.round(scale * tf.math.exp(-step / rate)), dtype=tf.int32)
    return call
                             
def const_batch_size(batch_size):
    def call(step):
        return batch_size*tf.ones_like(step)
    return call

def learning_rate_schedule(config, return_all=False):
    
    if config.lr_schedule is None:
        return Constant(0.0004)
    
    sched, start_lr, *params = config.lr_schedule
    
    s = {
        'constant': lambda params: Constant(start_lr),
        'exponential': lambda params: Exponential(config.n_steps, start_lr, *params),
        'linear': lambda params: Linear(config.n_steps, start_lr, *params),
    }
    
    if return_all:
        raise "I removed this functionality"
    
    schedule = s[sched](params)
    
    if config.lr_warmup:
        warmup_steps = config.lr_warmup
        schedule = Warmup(schedule, warmup_steps, start_lr)
    
    return schedule

def batch_size_schedule(config, name, params, return_all=False):
    s = {
        'exponential': exponential_batch_size(config.n_steps, *params)
    }
    if return_all:
        return s
    else:
        return s[name]

def show_schedules(config):
    lr_schedules = learning_rate_schedule(config, return_all=True)
    batch_size_schedules = batch_size_schedule(config, return_all=True)
    
    fig, axes = plt.subplots(len(lr_schedules), 1, figsize=(20,4*len(lr_schedules)))
    fig.suptitle("Learning Rate Schedules")
    x = tf.range(0, config.n_steps, 10, dtype=tf.float32)
    for ax, (name, lr_schedule) in zip(axes, lr_schedules.items()):
        ax.set_title(name)
        ax.plot(x, lr_schedule(x))
    fig, axes = plt.subplots(len(batch_size_schedules), 1, figsize=(20,4*len(batch_size_schedules)))
    fig.suptitle("Batch Size Schedules (incl minibatch size)")
    for ax, (name, bs_schedule) in zip(axes, batch_size_schedules.items()):
        if bs_schedule is None:
            continue
        ax.set_title(name)
        ax.plot(x, config['minibatch_size']*bs_schedule(x))
    plt.show()