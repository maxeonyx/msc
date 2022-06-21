import imp
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from . import util

from ml.data_bvh import np_dataset



class VizCallback(tf.keras.callbacks.Callback):
    def __init__(self, cfg, test_data_iter):
        super(VizCallback, self).__init__()
        self.cfg = cfg
        self.test_data_iter = test_data_iter

    def on_epoch_end(self, epoch, logs=None):
        for _ in range(2):
            x, y = next(self.test_data_iter)
            show_animations(self.cfg, x, self.model.predict(x, y, n_frames=self.cfg.predict_frames))
        plt.show()


def show_angles(cfg, ax, data):
    """
    Show animation as a reclustered image.
    Requires data be frame-aligned
    """

    data = tf.transpose(data, [1, 0])
    ax.imshow(data[:, :].numpy())


circular_means = None
def show_animations(cfg, inp_data, pred_data):
    """
    Show model input and output as reclustered images.
    Requires data be frame-aligned
    """
    global circular_means

    if circular_means is None:
        # pre-calculate circular means across the whole dataset for
        # consistent re-clustering vizualizations
        d = np_dataset()
        dd = np.concatenate(d[:, 1])
        circular_means = tf.cast(util.circular_mean(dd, axis=0), tf.float32)
        circular_means = circular_means[:cfg.n_hands, :cfg.n_dof]
        circular_means = tf.reshape(circular_means, [cfg.n_hands * cfg.n_dof])

    _fig, ax = plt.subplots(2)

    inp_data = inp_data["angles"]
    pred_data = pred_data

    def reshape_data(data):
        n_frames = data.shape[-1] // (cfg.n_hands * cfg.n_dof)
        data = tf.reshape(data, [n_frames, cfg.n_hands * cfg.n_dof])
        return data

    inp_data = reshape_data(inp_data)
    pred_data = reshape_data(pred_data)
    
    all_data = tf.concat([inp_data, pred_data], axis=0)
    all_data = util.recluster(all_data, frame_axis=0, circular_means=circular_means)

    inp_data = all_data[:inp_data.shape[0]]
    pred_data = all_data[inp_data.shape[0]:]

    show_angles(cfg, ax[0], inp_data)
    show_angles(cfg, ax[1], pred_data)
