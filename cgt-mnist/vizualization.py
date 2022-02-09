from sklearn.cluster import MiniBatchKMeans
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model, Input, layers
from IPython.display import display
import tensorflow_datasets as tfds
import time
import matplotlib.pyplot as plt
import enlighten
import tensorflow_probability as tfp
from dotmap import DotMap

import datasets

class Viz:
    
    def __init__(self, config, ds, centroids):
        
        self.config = config
        self.ds = ds
        self.centroids = centroids

    def np_showSeq(self, seq, size, max_images=3, cmap=None, return_fig=False):
        """ Show one or more images encoded as sequence. (numpy version)

            seq: numpy array of sequences which encode the image. Either a single sequence or multiple sequences.
            size: the image size. e.g. (28, 28) for `mnist` images.
            max_images: the maximum number of images to display.
        """ 
        batch = seq.shape[0]
        num_show_img = min(max_images, seq.shape[0])
        img = np.reshape(seq, (batch, *size, -1))

        fig=plt.figure(figsize=(3*num_show_img, 3))
        for i in range(num_show_img):
            ax = fig.add_subplot(1, num_show_img, i+1)
            ax.set_axis_off()
            plt.imshow(img[i], cmap=cmap)
        plt.show()
        if return_fig:
            return fig
    
    def scatter_on_bg(self, seq, idxs, output_length):
        batch_size = idxs.shape[0]
        seq_length = idxs.shape[1]
        n_color_dims = 3
                      
        color = tf.constant(self.config.bg_color or [1., 0., 1.])
        bg = tf.tile(color[None, None, :], [batch_size, output_length, 1])
        
        batch_idxs = tf.range(batch_size)
        color_idxs = tf.range(n_color_dims)
        batch_idxs = tf.tile(batch_idxs[:, None, None], [1, seq_length, n_color_dims])
        color_idxs = tf.tile(color_idxs[None, None, :], [batch_size, seq_length, 1])
        idxs = tf.tile(idxs[:, :, None], [1, 1, n_color_dims])
        idxs_nd = tf.stack([batch_idxs, idxs, color_idxs], axis=-1)

        seq = tf.tensor_scatter_nd_update(bg, idxs_nd, seq)
        
        return seq

    def showSeq(self, seq, idxs, size, max_images=3, cmap='gray', unshuffle=False, do_unquantize=True, return_fig=False):
        """ Show one or more images encoded as sequence. (tensorflow version)

            seq: tensor of sequences which encode the image. Either a single sequence or multiple sequences.
            size: the image size. e.g. (28, 28) for `mnist` images.
            max_images: the maximum number of images to display.
        """
        batch_size = idxs.shape[0]
        seq_length = idxs.shape[1]
        height, width = size
        img_length = height*width
        
        # unquantize to rgb
        if do_unquantize:
            seq = self.ds.unquantize(seq, to="rgb")
        elif seq.shape[-1] == 1:
            seq = self.ds.to_grayscale_rgb(seq)
        elif seq.shape[-1] != 3:
            print("seq.shape:", seq.shape)
            print("idxs.shape:", idxs.shape)
            assert False, "must either have a color dim with len 1 or 3, or use the do_unquantize flag"
        
        if unshuffle:
            seq = self.scatter_on_bg(seq, idxs, img_length)

        return self.np_showSeq(seq, size, max_images, cmap, return_fig=return_fig)

    def compare_quantized_and_unquantized(self, dataset_test_original):
        
        NUM_SAMPLES = 10
        
        ds_test = (
            dataset_test_original
            .map(datasets.normalize_image)
            .map(datasets.flatten)
            .map(self.ds.add_indices)
            .batch(NUM_SAMPLES)
        )
        ds_test_quantized = (
            dataset_test_original
            .map(datasets.normalize_image)
            .map(datasets.flatten)
            .map(self.ds.quantize)
            .map(self.ds.add_indices)
            .batch(NUM_SAMPLES)
        )
        
        print("quantized:")
        ex, ex_idxs, _ = next(iter(ds_test_quantized))
        self.showSeq(ex, ex_idxs, self.config.dataset.image_size, NUM_SAMPLES, unshuffle=False)
        print("unquantized:")
        ex, ex_idxs, _ = next(iter(ds_test))
        self.showSeq(ex, ex_idxs, self.config.dataset.image_size, NUM_SAMPLES, unshuffle=False, do_unquantize=False) # don't unquantize because it's not quantized
    
    def compare_shuffled_and_unshuffled(self, dataset_test_original):
        
        NUM_SAMPLES = 10

        ds_test_shuffled = (
            dataset_test_original
            .map(datasets.normalize_image)
            .map(datasets.flatten)
            .map(self.ds.quantize)
            .map(self.ds.shuffle_and_add_indices)
            .batch(NUM_SAMPLES)
        )
        print("shuffled:")
        ex, ex_idxs, _ = next(iter(ds_test_shuffled))
        self.showSeq(ex, ex_idxs, self.config.dataset.image_size, NUM_SAMPLES, unshuffle=False)
        print("unshuffled:")
        ex, ex_idxs, _ = next(iter(ds_test_shuffled))
        self.showSeq(ex, ex_idxs, self.config.dataset.image_size, NUM_SAMPLES, unshuffle=True)