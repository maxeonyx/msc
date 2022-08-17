import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

def von_mises_fisher_dist(p) -> tfd.VonMisesFisher:
    mean_direction = p[..., 0:2]
    concentration = p[..., 2]
    return tfd.VonMisesFisher(mean_direction=mean_direction, concentration=concentration)

def von_mises_fisher_sample(d):
    vecs = d.sample()
    angles = tf.atan2(vecs[..., 1], vecs[..., 0])
    return angles

def von_mises_fisher_mean(d):
    vecs = d.mean()
    angles = tf.atan2(vecs[..., 1], vecs[..., 0])
    return angles

def von_mises_fisher_loss(targets, p):
    d = von_mises_fisher_dist(p)
    sin = tf.math.sin(targets)
    cos = tf.math.cos(targets)
    targets = tf.stack([sin, cos], axis=-1)
    return -tf.reduce_mean(d.log_prob(targets))

def make_von_mises_fisher_decoder(cfg, name="vmf_decoder"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    
    params = layers.Dense(3, name="params")(inputs)
    mean_direction, _norm = tf.linalg.normalize(params[..., 0:2], axis=-1) # normalize mean_direction
    concentration = tf.nn.softplus(params[..., 2:])
    params = tf.concat([mean_direction, concentration], axis=-1)

    return Model(inputs=inputs, outputs=params, name=name)

def von_mises_fisher(cfg, name="von_mises_fisher"):
    vmf_decoder = make_von_mises_fisher_decoder(cfg, name=f"{name}_decoder")
    return von_mises_fisher_loss, von_mises_fisher_dist, von_mises_fisher_mean, von_mises_fisher_sample, vmf_decoder

def von_mises_dist(p):
    loc = p[: , :, 0]
    concentration = tf.exp(p[:, :, 1])
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

    return von_mises_loss, [von_mises_mean, von_mises_sample], Model(inputs=inputs, outputs=params, name=f"{name}_params")

def query_decoder(cfg, name="query_decoder"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")

    params = layers.Dense(2, name="params")(inputs)
    return Model(inputs=inputs, outputs=params, name=f"{name}_params")

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    ]
)
def angular_squared_error(targets, p):
    """
    Angular squared error between targets in [-pi, pi) and p in <[-1, 1), [-1, 1)>
    """
    target_sin = tf.sin(targets)
    target_cos = tf.cos(targets)
    p_sin = p[..., 0]
    p_cos = p[..., 1]
    
    return tf.square(target_sin - p_sin) + tf.square(target_cos - p_cos)

def to_angle(p):
    return tf.atan2(p[..., 1], p[..., 0])

def angular(cfg, name="angular"):
    inputs = Input(shape=[None, cfg.embd_dim], dtype=tf.float32, name="latents")
    params = layers.Dense(2, name="sincos")(inputs)
    decoder = Model(inputs=inputs, outputs=params, name=f"{name}_decoder")

    return angular_squared_error, [to_angle], decoder
