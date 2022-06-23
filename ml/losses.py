import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

class MSE:
    def __init__(self):
        self.name = "MSE"
        self.loss_fn = keras.losses.MeanSquaredError()
        self.unembed = keras.layers.Dense(1, name="unembed_" + self.name)
        self.sample_is_mean = True
    
    def loss(self, target, pred):
        # tf.print("Target:", target[0, :], "Pred:", pred[0, :])
        # tf.print()
        return self.loss_fn(target, pred)
    
    def sample(self, params):
        angles = params[..., 0]
        return angles

class MSE_angle:
    def __init__(self):
        self.name = "MSE"
        self.loss_fn = keras.losses.MeanSquaredError()
        self.unembed = keras.layers.Dense(2, name="unembed_" + self.name)
        self.sample_is_mean = True
    
    def loss(self, target, pred):
        target_sin = tf.math.sin(target)
        target_cos = tf.math.cos(target)
        target = tf.stack([target_sin, target_cos], axis=-1)
        return self.loss_fn(target, pred)
    
    def sample(self, params):
        sin = params[..., 0]
        cos = params[..., 1]
        angles = tf.math.atan2(sin, cos)
        return angles

class VonMises:

    def __init__(self):
        self.name = "VonMises"
        self.unembed_loc = keras.layers.Dense(1, name=self.name + "_unembed_loc")
        self.unembed_con = keras.layers.Dense(1, name=self.name + "_unembed_con")
        self.sample_is_mean = False
    
    def unembed(self, embd):
        loc = self.unembed_loc(embd)
        concentration = self.unembed_con(embd)
        concentration = tf.nn.relu(concentration)
        return tf.concat([loc, concentration], axis=-1)

    def loss(self, targets, pred_params):
        loc = pred_params[..., 0]
        concentration = pred_params[..., 1]
        dist = tfp.distributions.VonMises(loc=loc, concentration=concentration)

        loss = dist.log_prob(targets)
        
        return -tf.reduce_mean(loss)
    
    def sample(self, params):
        loc = params[..., 0]
        concentration = params[..., 1]
        dist = tfp.distributions.VonMises(loc=loc, concentration=concentration)
        return dist.sample()

    
    def mean(self, params):
        loc = params[..., 0]
        concentration = params[..., 1]
        dist = tfp.distributions.VonMises(loc=loc, concentration=concentration)
        return dist.mean()
