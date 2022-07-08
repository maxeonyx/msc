import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

def von_mises_fisher_dist(p):
    mean_direction, _norm = tf.linalg.normalize(p[..., 0:2], axis=-1) # normalize mean_direction
    concentration = tf.nn.softplus(p[..., 2])
    return tfd.VonMisesFisher(mean_direction=mean_direction, concentration=concentration)

def von_mises_fisher_loss(targets, p):
    d = von_mises_fisher_dist(p)
    sin = tf.math.sin(targets)
    cos = tf.math.cos(targets)
    targets = tf.stack([sin, cos], axis=-1)
    return -tf.reduce_mean(d.log_prob(targets))

def von_mises_fisher(cfg, name="von_mises_fisher"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    params = layers.Dense(3, name="params")(inputs)

    return von_mises_fisher_loss, von_mises_fisher_dist, Model(inputs=inputs, outputs=params, name=f"{name}_params")

def von_mises_dist(p):
    loc = p[: , :, 0]
    concentration = tf.exp(p[:, :, 1])
    return tfd.VonMises(loc=loc, concentration=concentration)

def von_mises_loss(targets, p):
    d = von_mises_dist(p)
    return -tf.reduce_mean(d.log_prob(targets))

def von_mises(cfg, name="von_mises"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    params = layers.Dense(2, name="params")(inputs)

    return von_mises_loss, von_mises_dist, Model(inputs=inputs, outputs=params, name=f"{name}_params")
