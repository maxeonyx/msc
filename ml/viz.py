import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class VizCallback(tf.keras.callbacks.Callback):
    def __init__(self, cfg, test_data):
        super(VizCallback, self).__init__()
        self.cfg = cfg
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        show_animations(self.cfg, self.model.predict(self.test_data, start_frame=self.cfg.chunk_size, n_frames=self.cfg.predict_frames))

def show_animations(cfg, data):
    """
    Show animation as 2 reclustered images
    """
    n_frames = data.shape[-1] // (cfg.n_hands * cfg.n_dof)
    data = tf.reshape(data, [n_frames, cfg.n_hands * cfg.n_dof])
    data = tf.transpose(data, [1, 0])

    fig, ax = plt.subplots()
    ax.imshow(data[:, :].numpy())
    # ax[1].imshow(data[:, :].numpy())
    plt.show()
