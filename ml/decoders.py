import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

def von_mises_fisher_dist(p):
    mean_direction, _norm = tf.linalg.normalize(p[..., 0:2], axis=-1) # normalize mean_direction
    concentration = tf.nn.softplus(p[..., 2])
    return tfd.VonMisesFisher(mean_direction=mean_direction, concentration=concentration)

def von_mises_fisher_sample(d):
    vecs = d.sample()
    angles = tf.atan2(vecs[..., 1], vecs[..., 0])
    return angles

def von_mises_fisher_mean(d):
    vecs = d.mean()
    angles = tf.atan2(vecs[..., 1], vecs[..., 0])
    return angles

@tf.function(jit_compile=True, input_signature=[
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
])
def von_mises_fisher_loss(targets, p):
    d = von_mises_fisher_dist(p)
    sin = tf.math.sin(targets)
    cos = tf.math.cos(targets)
    targets = tf.stack([sin, cos], axis=-1)
    return -tf.reduce_mean(d.log_prob(targets))

def von_mises_fisher(cfg, name="von_mises_fisher"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    params = layers.Dense(3, name="params")(inputs)

    return von_mises_fisher_loss, von_mises_fisher_dist, von_mises_fisher_mean, von_mises_fisher_sample, Model(inputs=inputs, outputs=params, name=f"{name}_params")

def von_mises_dist(p):
    loc = p[: , :, 0]
    concentration = tf.exp(p[:, :, 1])
    return tfd.VonMises(loc=loc, concentration=concentration)

def von_mises_loss(targets, p):
    d = von_mises_dist(p)
    return -tf.reduce_mean(d.log_prob(targets))

def von_mises_mean(d):
    return d.mean()

def von_mises_sample(d):
    return d.sample()

def von_mises(cfg, name="von_mises"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    params = layers.Dense(2, name="params")(inputs)

    return von_mises_loss, von_mises_dist, von_mises_mean, von_mises_sample, Model(inputs=inputs, outputs=params, name=f"{name}_params")

def query_decoder(cfg, name="query_decoder"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")

    

    params = layers.Dense(2, name="params")(inputs)
    return Model(inputs=inputs, outputs=params, name=f"{name}_params")
