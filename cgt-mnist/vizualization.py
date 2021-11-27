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

    def showSeq(self, seq, idxs, size, max_images=3, cmap='gray', unshuffle=False, do_unquantize=True):
        """ Show one or more images encoded as sequence. (tensorflow version)

            seq: tensor of sequences which encode the image. Either a single sequence or multiple sequences.
            size: the image size. e.g. (28, 28) for `mnist` images.
            max_images: the maximum number of images to display.
        """
        batch_size = idxs.shape[0]
        seq_length = idxs.shape[1]
        img_length = size[0]*size[1]
        if unshuffle:
            batch_idxs = tf.tile(tf.expand_dims(tf.range(batch_size), -1), [1, seq_length])
            idxs_nd = tf.concat([tf.expand_dims(batch_idxs, -1), tf.expand_dims(idxs, -1)], axis=-1)
            seq = tf.scatter_nd(idxs_nd, seq, [batch_size, img_length])

        if do_unquantize:
            seq = tf.map_fn(fn=self.ds.unquantize, elems=seq, fn_output_signature=tf.float16)
        seq = tf.cast(seq, float).numpy()

        return self.np_showSeq(seq, size, max_images, cmap)

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