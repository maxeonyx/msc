import io

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from ml import data_bvh

from . import util

from ml.data_bvh import np_dataset

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image



class VizCallback(tf.keras.callbacks.Callback):
    def __init__(self, cfg, test_data_iter, log_dir):
        super(VizCallback, self).__init__()
        self.cfg = cfg
        self.test_data_iter = test_data_iter
        self.log_dir = log_dir
        self.test_inputs = [next(test_data_iter) for _ in range(cfg.n_test_samples)]

    def on_epoch_end(self, epoch, logs=None):
        writer = tf.summary.create_file_writer(self.log_dir)
        with writer.as_default():
            for i, (x, y) in enumerate(self.test_inputs):
                y_pred_samples, y_pred_mean = self.model.predict(x, y, n_frames=self.cfg.predict_frames)
                img = plot_to_image(show_animations(self.cfg, y, y_pred_samples, y_pred_mean))
                tf.summary.image(f"anim_track_{i+1}", img, step=epoch)
        writer.close()


def show_angles(cfg, ax, data):
    """
    Show animation as a reclustered image.
    Requires data be frame-aligned
    """

    data = tf.transpose(data, [1, 0])
    ax.imshow(data[:, :].numpy())


def show_animations(cfg, tar_data, pred_data, pred_mean):
    """
    Show model input and output as reclustered images.
    Requires data be frame-aligned
    """
        
    fig, ax = plt.subplots(3 if pred_mean is not None else 2)

    tar_data = tar_data["angles"]
    pred_data = pred_data

    def reshape_data(data):
        n_frames = data.shape[-1] // (cfg.n_hands * cfg.n_dof)
        data = tf.reshape(data, [n_frames, cfg.n_hands * cfg.n_dof])
        return data

    tar_data = reshape_data(tar_data)
    pred_data = reshape_data(pred_data)
    if pred_mean is not None:
        pred_mean = reshape_data(pred_mean)

    show_angles(cfg, ax[0], tar_data)
    show_angles(cfg, ax[1], pred_data)
    if pred_mean is not None:
        show_angles(cfg, ax[2], pred_mean)

    return fig
