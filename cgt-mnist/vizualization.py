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

    def np_showSeq(self, seq, size, max_images=3, cmap=None):
        """ Show one or more images encoded as sequence. (numpy version)

            seq: numpy array of sequences which encode the image. Either a single sequence or multiple sequences.
            size: the image size. e.g. (28, 28) for `mnist` images.
            max_images: the maximum number of images to display.
        """ 
        batch = seq.shape[0]
        num_show_img = min(max_images, seq.shape[0])
        img = np.reshape(seq, (batch, *size, -1))
        if img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)    

        fig=plt.figure(figsize=(3*num_show_img, 3))
        for i in range(num_show_img):
            ax = fig.add_subplot(1, num_show_img, i+1)
            ax.set_axis_off()
            plt.imshow(img[i], cmap=cmap)
        plt.show()
        return fig
    
    def unquantize(self, seq):
        seq = tf.map_fn(fn=self.ds.unquantize, elems=seq, fn_output_signature=tf.float32)
        return tf.squeeze(seq, axis=-1)
    
    def scatter_on_bg(self, seq, idxs, output_length):
        batch_size = idxs.shape[0]
        seq_length = idxs.shape[1]
                      
        color = tf.constant(self.config.bg_color or [1., 0., 1.])
        bg = tf.tile(color[None, None, :], [batch_size, output_length, 1])

        # convert seq to grayscale rgb
        seq = tf.tile(seq[:, :, None], [1, 1, 3])

        batch_idxs = tf.tile(tf.expand_dims(tf.range(batch_size), -1), [1, seq_length])
        idxs_nd = tf.concat([tf.expand_dims(batch_idxs, -1), tf.expand_dims(idxs, -1)], axis=-1)

        seq = tf.tensor_scatter_nd_update(bg, idxs_nd, seq)
        
        return seq

    def showSeq(self, seq, idxs, size, max_images=3, cmap='gray', unshuffle=False, do_unquantize=True):
        """ Show one or more images encoded as sequence. (tensorflow version)

            seq: tensor of sequences which encode the image. Either a single sequence or multiple sequences.
            size: the image size. e.g. (28, 28) for `mnist` images.
            max_images: the maximum number of images to display.
        """
        print("seq shape:", seq.shape)
        batch_size = idxs.shape[0]
        seq_length = idxs.shape[1]
        img_length = size[0]*size[1]
        if do_unquantize:
            seq = self.unquantize(seq)
        else:
            seq = tf.cast(seq, tf.float32)
        print("seq shape:", seq.shape)
        if unshuffle:
            seq = self.scatter_on_bg(seq, idxs, img_length)

        return self.np_showSeq(seq, size, max_images, cmap)

    def showSeqExpectedVal(self, seq, probs, idxs, size, max_images=3, cmap='gray', unshuffle=False):
        """ Show one or more images encoded as sequence. (tensorflow version)

            seq: tensor of sequences which encode the image. Either a single sequence or multiple sequences.
            size: the image size. e.g. (28, 28) for `mnist` images.
            max_images: the maximum number of images to display.
        """
        batch_size = idxs.shape[0]
        seq_length = idxs.shape[1]
        img_length = size[0]*size[1]
        
        # do_unquantize for samples first
        seq = tf.map_fn(fn=self.ds.unquantize, elems=seq, fn_output_signature=tf.float32)
        
        centroids = tf.reshape(self.centroids, [1, 1, -1])
        expected_col = tf.tensordot(probs, centroids, axes=([2], [2]))
        expected_col = tf.squeeze(expected_col, axis=-1)
        
        seq = tf.concat([seq, expected_col], axis=1)
        seq = tf.squeeze(seq)
        return self.showSeq(seq, idxs, size, max_images, cmap, unshuffle=unshuffle, do_unquantize=False)

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
        self.showSeq(ex, ex_idxs, (self.config['image_width'], self.config['image_height']), NUM_SAMPLES, unshuffle=False)
        print("unquantized:")
        ex, ex_idxs, _ = next(iter(ds_test))
        self.showSeq(ex, ex_idxs, (self.config['image_width'], self.config['image_height']), NUM_SAMPLES, unshuffle=False, do_unquantize=False) # don't unquantize because it's not quantized
    
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
        self.showSeq(ex, ex_idxs, (self.config['image_width'], self.config['image_height']), NUM_SAMPLES, unshuffle=False)
        print("unshuffled:")
        ex, ex_idxs, _ = next(iter(ds_test_shuffled))
        self.showSeq(ex, ex_idxs, (self.config['image_width'], self.config['image_height']), NUM_SAMPLES, unshuffle=True)