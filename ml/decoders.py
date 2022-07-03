import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

def von_mises_fisher(cfg, name="von_mises_fisher"):

    def dist(params):
        mean_direction = tf.linalg.normalize(params[..., 0:2], axis=-1) # normalize mean_direction
        concentration = tf.nn.softplus(params[..., 2])
        return tfd.VonMisesFisher(mean_direction=mean_direction, concentration=concentration)

    def loss(d, targets):
        sin = tf.math.sin(targets)
        cos = tf.math.cos(targets)
        targets = tf.stack([sin, cos], axis=-1)
        return -tf.reduce_mean(d.log_prob(targets))
    
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    params = layers.Dense(3, name="params")(inputs)
    d = tfp.layers.DistributionLambda(dist)(params)

    return loss, Model(inputs=inputs, outputs=d, name=f"{name}_decoder")

def von_mises(cfg, name="von_mises"):

    def dist(params):
        loc = params[..., 0]
        concentration = tf.nn.softplus(params[..., 1])
        return tfd.VonMises(loc=loc, concentration=concentration)
    
    def loss(d, targets):
        return -tf.reduce_mean(d.log_prob(targets))

    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    params = layers.Dense(2, name="params")(inputs)
    d = tfp.layers.DistributionLambda(dist)(params)

    return loss, Model(inputs=inputs, outputs=d, name=f"{name}_decoder")
