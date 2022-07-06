from tensorflow import keras
import tensorflow as tf


def angle_wrap(angles):
    angles = tf.math.atan2(tf.sin(angles), tf.cos(angles))
    # angles = tf.where(angles < -np.pi, angles+np.pi*2, angles)
    # angles = tf.where(angles > np.pi, angles-np.pi*2, angles)
    return angles

def circular_mean(angles, axis=0):
    # compute the circular mean of the data for this example+track
    # rotate the data so that the circular mean is 0
    # store the circular mean
    means_cos_a = tf.reduce_mean(tf.math.cos(angles), axis=axis)
    means_sin_a = tf.reduce_mean(tf.math.sin(angles), axis=axis)
    circular_means = tf.math.atan2(means_sin_a, means_cos_a)
    return circular_means

def recluster(angles, frame_axis=0, circular_means=None):
    if circular_means is None: 
        # rotate the data so the circular mean is 0
        circular_means = circular_mean(angles, axis=frame_axis)
    angles = angles - tf.expand_dims(circular_means, axis=frame_axis)
    angles = angle_wrap(angles)
    return angles

class WarmupLRSchedule(keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, peak_learning_rate, warmup_steps, other_schedule=None):
        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps
        self.other_schedule = other_schedule
        if other_schedule is None:
            self.other_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=peak_learning_rate,
                decay_steps=1000,
                decay_rate=0.96
            )

    def __call__(self, step):
        return tf.cond(
            pred=tf.less(step, self.warmup_steps),
            true_fn=lambda: self.peak_learning_rate * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32),
            false_fn=lambda: self.other_schedule(step)
        )
    
    def get_config(self):
        return {
            'peak_learning_rate': self.peak_learning_rate,
            'warmup_steps': self.warmup_steps,
            'other_schedule': self.other_schedule
        }

class KerasLossWrapper(tf.keras.metrics.Metric):
    """
    Show instantaneous loss in keras progress bar.
    """

    def __init__(self, loss_fn, name="instantaneous_loss", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.loss_fn = loss_fn
        self.total = self.add_weight(name="total", initializer="zeros")
    
    def reset_state(self):
        pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = self.loss_fn(y_true, y_pred)
        self.total.assign(loss)

    def result(self):
        return self.total
