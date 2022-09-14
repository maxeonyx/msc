import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from math import pi, tau

from ml import utils

def von_mises_fisher_dist(p) -> tfd.VonMisesFisher:
    mean_direction, _norm = tf.linalg.normalize(p[..., 0:2], axis=-1) # normalize mean_direction
    concentration = tf.nn.softplus(p[..., 2])
    return tfd.VonMisesFisher(mean_direction=mean_direction, concentration=concentration)

@tf.autograph.experimental.do_not_convert
def von_mises_fisher_sample(p):
    d = von_mises_fisher_dist(p)
    sincos = d.sample()
    return tf.atan2(sincos[..., 0], sincos[..., 1])

def von_mises_fisher_mean(p):
    d = von_mises_fisher_dist(p)
    sincos = d.mean()
    return tf.atan2(sincos[..., 0], sincos[..., 1])

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    ]
)
def von_mises_fisher_loss(targets, p):
    d = von_mises_fisher_dist(p)
    sin = tf.math.sin(targets)
    cos = tf.math.cos(targets)
    targets = tf.stack([sin, cos], axis=-1)
    return -tf.reduce_mean(d.log_prob(targets))

def von_mises_fisher(cfg, name="von_mises_fisher"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    params = layers.Dense(3, name="params")(inputs)
    vmf_decoder = Model(inputs=inputs, outputs=params, name=name)

    return von_mises_fisher_loss, vmf_decoder, {
        "mean": von_mises_fisher_mean,
        "sample": von_mises_fisher_sample,
    }

def von_mises_atan_dist(p):
    mean_direction, _norm = tf.linalg.normalize(p[..., 0:2], axis=-1) # normalize mean_direction
    loc = tf.math.atan2(mean_direction[..., 0], mean_direction[..., 1])
    concentration = tf.nn.softplus(p[..., 2])
    return tfd.VonMises(loc=loc, concentration=concentration)

def von_mises_atan_loss(targets, p):
    d = von_mises_atan_dist(p)
    return -tf.reduce_mean(d.log_prob(targets))

def von_mises_atan_mean(p):
    d = von_mises_atan_dist(p)
    return d.mean()

def von_mises_atan_sample(p):
    d = von_mises_atan_dist(p)
    return d.sample()

def von_mises_atan(cfg, name="von_mises_atan"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    params = layers.Dense(3, name="params")(inputs)
    angular_decoder = Model(inputs=inputs, outputs=params, name=f"{name}_params")

    return von_mises_atan_loss, angular_decoder, {
        "mean": von_mises_atan_mean,
        "sample": von_mises_atan_sample
    }


def von_mises_dist(p):
    loc = utils.angle_wrap(p[..., 0])
    concentration = tf.nn.softplus(p[..., 1])
    return tfd.VonMises(loc=loc, concentration=concentration)

def von_mises_loss(targets, p):
    d = von_mises_dist(p)
    return -tf.reduce_mean(d.log_prob(targets))

def von_mises_mean(p):
    d = von_mises_dist(p)
    return d.mean()

def von_mises_sample(p):
    d = von_mises_dist(p)
    return d.sample()

def von_mises(cfg, name="von_mises"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    params = layers.Dense(2, name="params")(inputs)
    angular_decoder = Model(inputs=inputs, outputs=params, name=f"{name}_params")

    return von_mises_loss, angular_decoder, {
        "mean": von_mises_mean,
        "sample": von_mises_sample
    }

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    ]
)
def angular_squared_error(targets: tf.Tensor, p: tf.Tensor):
    """
    Angular squared error between targets in [-pi, pi) and p in <[-1, 1), [-1, 1)>
    """
    target_sin = tf.sin(targets)
    target_cos = tf.cos(targets)
    p_sin = p[..., 0]
    p_cos = p[..., 1]
    
    return tf.square(target_sin - p_sin) + tf.square(target_cos - p_cos)

def to_angle(p):
    return tf.atan2(p[..., 0], p[..., 1])

def angular(cfg, name="angular"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    params = layers.Dense(2, name="sincos")(inputs)
    decoder = Model(inputs=inputs, outputs=params, name=f"{name}_decoder")

    return angular_squared_error, decoder, {
        "mean": to_angle,
    }

def categorical(cfg, name="categorical"):

    def to_categorical(angle):
        """
        Convert an angle in the range [-pi, pi) to a one-hot vector
        with shape [cfg.n_categories]
        """
        # convert to range [0, 1)
        angle = (angle + pi) / tau
        # convert to int in range [0, cfg.n_categories)
        angle = tf.cast(tf.math.floor(angle * cfg.n_categories), tf.int32)
        # convert to one-hot
        return tf.one_hot(angle, cfg.n_categories)

    def from_categorical(p):
        """
        Convert a one-hot vector with shape [cfg.n_categories] to an angle
        in the range [-pi, pi)
        """
        # convert to int in range [0, cfg.n_categories)
        p = tf.math.argmax(p, axis=-1)
        # convert to range [0, 1)
        p = tf.cast(p, tf.float32) / cfg.n_categories
        # convert to range [-pi, pi)
        p = p * tau - pi
        # ensure in range [-pi, pi)
        return utils.angle_wrap(p)


    def categorical_dist(p):
        return tfd.OneHotCategorical(logits=p)

    def categorical_loss(targets, p):
        d = categorical_dist(p)
        targets = to_categorical(targets)
        return -tf.reduce_mean(d.log_prob(targets))
    
    def categorical_mode(p):
        d = categorical_dist(p)
        return from_categorical(d.mode())
    
    def categorical_sample(p):
        d = categorical_dist(p)
        return from_categorical(d.sample())

    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    logits = layers.Dense(cfg.n_categories, name="logits")(inputs)
    decoder = Model(inputs=inputs, outputs=logits, name=f"{name}_decoder")

    return categorical_loss, decoder, {
        "mode": categorical_mode,
        "sample": categorical_sample,
    }


def get_prediction_head(cfg, typ):
    if typ == "angular_mse":
        return angular(cfg)
    elif typ == "vmf_crossentropy":
        return von_mises_fisher(cfg, name="vmf")
    elif typ == "vm_atan_crossentropy":
        return von_mises_atan(cfg, name="vm_atan")
    elif typ == "vm_crossentropy":
        return von_mises(cfg, name="vm")
    elif typ == "categorical":
        return categorical(cfg, name="categorical")
    else:
        raise ValueError("Unknown loss type '{}'".format(typ))
