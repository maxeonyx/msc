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

class WarmupInvSquare(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embd_dim, warmup_steps):
        super(WarmupInvSquare, self).__init__()

        self.d_model = embd_dim
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = step * (self.warmup_steps ** -1.5)
        arg2 = tf.math.rsqrt(step)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class WarmupLinear(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, peak_lr, warmup_steps):
        super(WarmupLinear, self).__init__()
        
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = step/self.warmup_steps
        arg2 = 1 - (step-self.warmup_steps)/(self.total_steps - self.warmup_steps)
        
        return self.peak_lr * tf.math.minimum(arg1, arg2)

class Constant(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, rate):
        super(Constant, self).__init__()
        self.rate = rate
    
    def __call__(self, step):
        return tf.ones_like(step)*self.rate

class Warmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, warmup_steps, end_lr):
        
    
class WarmupExponential(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, warmup_steps, peak_lr, final_lr):
        super(WarmupExponential, self).__init__()
        self.scale = tf.cast(peak_lr, tf.float64)
        self.rate = tf.cast(total_steps/tf.math.log(peak_lr/final_lr), tf.float64)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float64)
        decay_step = tf.cast(step-self.warmup_steps, tf.float64)
        arg1 = step/self.warmup_steps
        arg2 = tf.math.exp(-decay_step / self.rate)
        return self.scale * tf.math.minimum(arg1, arg2)


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
        'constant': Constant(max_lr),
        'exponential': Exponential(config.n_steps, start_lr, *params),
        'linear': Linear(config.n_steps, start_lr, *params),
    }
    
    if return_all:
        return s
    
    schedule = s[sched]
    
    if config.lr_warmup is not None:
        warmup_steps = config.lr_warmup
        schedule = Warmup(warmup_steps, start_lr

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