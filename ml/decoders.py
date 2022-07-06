import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

def von_mises_fisher_dist(p):
    return tfd.VonMisesFisher(mean_direction=p[:, :, :2], concentration=p[:, :, 2])

def von_mises_fisher_loss(targets, p):
    d = von_mises_fisher_dist(p)
    sin = tf.math.sin(targets)
    cos = tf.math.cos(targets)
    targets = tf.stack([sin, cos], axis=-1)
    return -tf.reduce_mean(d.log_prob(targets))

def von_mises_fisher(cfg, name="von_mises_fisher"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    params = layers.Dense(3, name="params")(inputs)
    mean_direction, _norm = tf.linalg.normalize(params[..., 0:2], axis=-1) # normalize mean_direction
    concentration = tf.nn.softplus(params[..., 2:])
    params = tf.concat([mean_direction, concentration], axis=-1)
    d = tfp.layers.DistributionLambda(von_mises_fisher_dist)(params)

    return von_mises_fisher_loss, von_mises_dist, Model(inputs=inputs, outputs=d, name=f"{name}_params")

def von_mises_dist(p):
    loc = p[: , :, 0]
    concentration = tf.nn.softplus(p[:, :, 1])
    return tfd.VonMises(loc=loc, concentration=concentration)

def von_mises_loss(targets, p):
    d = von_mises_dist(p)
    return -tf.reduce_mean(d.log_prob(targets))

def von_mises(cfg, name="von_mises"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    params = layers.Dense(2, name="params")(inputs)

    return von_mises_loss, von_mises_dist, Model(inputs=inputs, outputs=params, name=f"{name}_params")
