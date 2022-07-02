import abc

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

class PredictionHead(keras.layers.Layer, abc.ABC):

    def __init__(self, name, is_distribution=False):
        super().__init__(name=name)
        self.is_distribution = is_distribution
    
    def call(self, inputs):
        return self.unembed(inputs)
    
    @abc.abstractmethod
    def unembed(self, embd):
        """
        Project out of the latent space into the format for the prediction head.
        """
        pass

    @abc.abstractmethod
    def loss(self, target, pred):
        """
        Compute the loss.
        """
        pass

    @abc.abstractmethod
    def mean(self, params):
        pass


class MSE(PredictionHead):
    """
    MSE with angles learned as scalars in the (-pi, pi] range.
    """
    def __init__(self):
        super().__init__("MSE")
        self.loss_fn = keras.losses.MeanSquaredError()
        self.unembed_layer = keras.layers.Dense(1, name="unembed_" + self.name)
    
    def unembed(self, embd):
        return self.unembed_layer(embd)
    
    def loss(self, target, pred):
        angles = pred[..., 0]
        return self.loss_fn(target, angles)
    
    def mean(self, params):
        angles = params[..., 0]
        return angles


class MSE_angle(PredictionHead):
    """
    MSE with angles learned as (sin, cos) pairs
    """
    def __init__(self):
        super().__init__("MSE")
        self.loss_fn = keras.losses.MeanSquaredError()
        self.unembed_layer = keras.layers.Dense(2, name="unembed_" + self.name)
        self.sample_is_mean = True
    
    def unembed(self, embd):
        return self.unembed_layer(embd)
    
    def loss(self, target, pred):
        target_sin = tf.math.sin(target)
        target_cos = tf.math.cos(target)
        target = tf.stack([target_sin, target_cos], axis=-1)
        return self.loss_fn(target, pred)
    
    def mean(self, params):
        sin = params[..., 0]
        cos = params[..., 1]
        angles = tf.math.atan2(sin, cos)
        return angles


class DistBase(PredictionHead, abc.ABC):
    """
    Prediction head base class for distributional prediction heads.
    """

    def __init__(self, name):
        super().__init__(name, is_distribution=True)

    @abc.abstractmethod
    def dist(self, params):
        pass

    @abc.abstractmethod
    def sample(self, params):
        pass


class VonMises(DistBase):
    """
    Distribution defined over the range (-pi, pi]
    Angles are learned as scalars in the range (-pi, pi]
    """

    def __init__(self):
        super().__init__("VonMises")
        self.unembed_loc = keras.layers.Dense(1, name=self.name + "_unembed_loc")
        self.unembed_con = keras.layers.Dense(1, name=self.name + "_unembed_con")
    
    def unembed(self, embd):
        loc = self.unembed_loc(embd)
        concentration = self.unembed_con(embd)
         # concentration must be strictly positive, range (0, +inf)
        concentration = tf.nn.softplus(concentration)
        return tf.concat([loc, concentration], axis=-1)

    def dist(self, params):
        loc = params[..., 0]
        concentration = params[..., 1]
        return tfp.distributions.VonMises(loc=loc, concentration=concentration)

    def loss(self, targets, pred_params):
        d = self.dist(pred_params)
        log_lik = d.log_prob(targets)
        loss = -tf.reduce_mean(log_lik)
        return loss

    def mean(self, params):
        d = self.dist(params)
        return d.mean()
    
    def sample(self, params):
        d = self.dist(params)
        return d.sample()


class VonMisesFisher(DistBase):
    """
    Distribution defined over vectors on the unit circle.

    Angles are learned as (sin, cos) pairs.
    """

    def __init__(self):
        super().__init__("VonMisesFisher")
        self.unembed_mean = keras.layers.Dense(2, name=self.name + "_unembed_mean")
        self.unembed_conc = keras.layers.Dense(1, name=self.name + "_unembed_conc")
    
    def unembed(self, embd):
        mean_direction = self.unembed_mean(embd)
        mean_direction = tf.linalg.normalize(mean_direction, axis=-1) # normalize mean_direction
        concentration = self.unembed_conc(embd)
         # concentration must be strictly positive, range (0, +inf)
        concentration = tf.nn.softplus(concentration)
        return tf.concat([mean_direction, concentration], axis=-1)

    def dist(self, params):
        mean_direction = params[..., 0:2]
        concentration = params[..., 2]
        return tfp.distributions.VonMisesFisher(mean_direction=mean_direction, concentration=concentration)

    def loss(self, targets, pred_params):
        d = self.dist(pred_params)

        sin = tf.math.sin(targets)
        cos = tf.math.cos(targets)
        targets = tf.stack([sin, cos], axis=-1)

        log_lik = d.log_prob(targets)
        loss = -tf.reduce_mean(log_lik)
        return loss

    def mean(self, params):
        d = self.dist(params)
        vectors = d.mean()
        angles = tf.math.atan2(vectors[..., 0], vectors[..., 1])
        return angles
    
    def sample(self, params):
        d = self.dist(params)
        vectors = d.sample()
        angles = tf.math.atan2(vectors[..., 0], vectors[..., 1])
        return angles


class KerasLossWrapper(tf.keras.metrics.Metric):
    """
    Show instantaneous loss in keras progress bar.
    """

    def __init__(self, loss_fn, name=None, dtype=None):
        super().__init__(name="instantaneous_loss", dtype=dtype)
        self.loss_fn = loss_fn
        self.total = self.add_weight(name="total", initializer="zeros")
    
    def reset_state(self):
        pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = self.loss_fn(y_true, y_pred)
        self.total.assign(loss)

    def result(self):
        return self.total
