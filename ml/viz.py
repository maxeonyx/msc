import io
import math
from einops import rearrange

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from ml import data_bvh, util

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
  return image



class VizCallback(tf.keras.callbacks.Callback):
    def __init__(self, cfg, test_data_iter, log_dir):
        super(VizCallback, self).__init__()
        self.writer = tf.summary.create_file_writer(log_dir)
        self.cfg = cfg
        self.test_data_iter = test_data_iter
        self.test_inputs = next(test_data_iter)

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            x_batch, y_batch = self.test_inputs
            y_pred_mean_batch, y_pred_sample_batch = self.model.predict(x_batch, n_frames=self.cfg.predict_frames)
            imgs = []

            if y_pred_sample_batch is None:
                y_pred_sample_batch = [None for _ in range(y_pred_mean_batch.shape[0])]

            for x, y, y_pred_mean, y_pred_sample in zip(x_batch, y_batch, y_pred_mean_batch, y_pred_sample_batch):
                img = plot_to_image(show_animations(self.cfg, [y, y_pred_mean, y_pred_sample]))
                imgs.append(img)
            img = rearrange(imgs, "b h w c -> b h w c")
            tf.summary.image(f"anim_tracks", img, step=epoch*self.cfg.steps_per_epoch)
        


def show_angles(cfg, ax, data):
    """
    Show animation as a reclustered image.
    Requires data be frame-aligned
    """

    data = tf.transpose(data, [1, 0])
    ax.imshow(data[:, :].numpy(), vmin=-math.pi/2, vmax=math.pi/2, cmap="RdBu")


def show_animations(cfg, data):
    """
    Show model input and output as reclustered images.
    Requires data be frame-aligned
    """
    
    num_figs = len(data)

    fig, ax = plt.subplots(num_figs)

    def reshape_data(data):
        n_frames = data.shape[-1] // (cfg.n_hands * cfg.n_dof)
        data = tf.reshape(data, [n_frames, cfg.n_hands * cfg.n_dof])
        return data

    for i, d in enumerate(data):
        d = reshape_data(d)
        show_angles(cfg, ax[i], d)

    return fig
