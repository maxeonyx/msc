import numpy as np
import tensorflow as tf

def circular_mean(angles, axis=0):
    # compute the circular mean of the data for this example+track
    # rotate the data so that the circular mean is 0
    # store the circular mean
    means_cos_a = tf.reduce_mean(tf.math.cos(angles), axis=axis)
    means_sin_a = tf.reduce_mean(tf.math.sin(angles), axis=axis)
    circular_means = tf.math.atan2(means_sin_a, means_cos_a)
    return circular_means

def recluster(angles, frame_axis=0):
    # rotate the data so the circular mean is 0
    circular_means = circular_mean(angles, axis=frame_axis)
    angles = angles - tf.expand_dims(circular_means, axis=frame_axis)
    angles = tf.where(angles < -np.pi, angles+np.pi*2, angles)
    angles = tf.where(angles > np.pi, angles-np.pi*2, angles)
    return angles
