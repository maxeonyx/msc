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

def normalize_image(image, label):
    return tf.cast(image, dtype=tf.float32) / 255.0, label

def find_centroids(ds_train, num_clusters, batch_size):
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=batch_size, verbose=True)
    ds_batched = ds_train.map(normalize_image).batch(batch_size)
    with enlighten.get_manager() as manager:
        title = manager.status_bar(f"K-Means clustering to make {num_clusters}-color MNIST Dataset", justify=enlighten.Justify.CENTER)
        clusters_names = manager.status_bar(''.join('{:<10}'.format(f"cen. {i}") for i in range(num_clusters)))
        clusters_status = manager.status_bar(''.join('{:<10}'.format('??????') for _ in range(num_clusters)))
        pbar = manager.counter(total=60000//batch_size, desc='Discretize to 8 colors', unit='minibatches')
        for img, _ in pbar(iter(ds_batched)):
            pixels = img.numpy().reshape(-1, 1)
            kmeans.partial_fit(pixels)
            clusters_status.update(''.join('{:<10.3f}'.format(x[0]) for x in np.sort(kmeans.cluster_centers_, axis=0)))

        centroids = kmeans.cluster_centers_
        centroids = tf.convert_to_tensor(np.sort(centroids, axis=0), dtype=tf.float32)
        return centroids

def mnist_gamma_distribution():
    alpha, beta = 1.2, 0.007
    gamma = tfp.distributions.Gamma(concentration=alpha, rate=beta)
    return gamma, f"Gamma (alpha={alpha}, beta={beta})"
    
def plot_distribution(config, dist, name):

    def plot_cdf():
        x = tf.range(config['seq_length'], dtype=tf.float32)
        cdf = dist.cdf(x)
        fig, ax = plt.subplots()
        ax.plot(tf.range(config['seq_length']), cdf)
        ax.set_title(f"{name} CDF, distribution of examples by number of pixels.")


    def plot_pdf():
        x = tf.range(config['seq_length'], dtype=tf.float32)
        pdf = dist.prob(x)
        fig, ax = plt.subplots()
        ax.plot(tf.range(config['seq_length']), pdf)
        ax.set_title(f"{name} PDF, distribution of examples by number of pixels.")


    def test_sample():

        return tf.cast(tf.math.minimum(tf.math.round(dist.sample(sample_shape=[100])), config['seq_length'] - 1), tf.int32)

    plot_pdf()
    plot_cdf()
    print("Example integers sampled from gamma distribution.")
    print(test_sample())

def squared_euclidean_distance(a, b):
    b = tf.transpose(b)        
    a2 = tf.math.reduce_sum(tf.math.square(a), axis=1, keepdims=True)
    b2 = tf.math.reduce_sum(tf.math.square(b), axis=0, keepdims=True)
    ab = tf.linalg.matmul(a, b)
    return a2 - 2 * ab + b2

def flatten(image, label):
    shape = tf.shape(image) # (height, width, color)
    sequence = tf.reshape(image, (-1, shape[2])) # (height * width, color)
    return sequence, label

class Datasets:
    
    def __init__(self, config, ds_train_orig, ds_test_orig, centroids, distribution):
        
        self.centroids = centroids
        self.config = config
        self.dist = distribution
        self.ds_train_orig = ds_train_orig
        self.ds_test_orig = ds_test_orig
    
    def quantize(self, sequence, label):
        d = squared_euclidean_distance(sequence, self.centroids) # (height * width, centroids)
        sequence = tf.math.argmin(d, axis=1, output_type=tf.int32)  # (height * width)
        return sequence, label

    def unquantize(self, x):
        x_one_hot = tf.cast(tf.one_hot(x, depth=len(self.centroids)), dtype=tf.float32)  # (seq, num_centroids)
        return tf.linalg.matmul(x_one_hot,self.centroids)  # (seq, num_features)

    def expected_col(self, probs):
        centroids = tf.reshape(self.centroids, [1, 1, -1])
        expected_col = tf.tensordot(probs, centroids, axes=([2], [2]))
        expected_col = tf.squeeze(expected_col, axis=-1)
        return expected_col
    
    def shuffle_and_add_indices(self, sequence, label):

        idxs = tf.range(self.config['seq_length'], dtype=tf.int32)
        idxs = tf.random.shuffle(idxs)

        sequence = tf.gather(sequence, idxs)

        return sequence, idxs, label


    def add_n_from_distribution(self, batch_sequences, batch_idxs, batch_labels):
        bs = tf.shape(batch_sequences)[0]
        # sample single integer between 0 and 783 from given distribution
        n = tf.cast(tf.math.round(self.dist.sample(sample_shape=[])), tf.int32)
        n = tf.clip_by_value(n, 1, self.config['seq_length'] - 1)
        
        if self.config.batch_size_schedule == 'dynamic':
            return batch_sequences, batch_idxs, n
        else:
            return batch_sequences, batch_idxs, tf.repeat(n, self.config.num_devices)

    def add_indices(self, sequence, label):

        idxs = tf.range(self.config['seq_length'], dtype=tf.int32)

        return sequence, idxs, label

    def make_datasets(self, kind):
        
        if kind == 'random_split_shuffled':
            return self.random_split_shuffled()
        else:
            assert False, f'Dataset type "{kind}" isnt valid'
    
    def random_split_shuffled(self):
        
        dataset_train_shuf = (
            self.ds_train_orig
            .map(normalize_image)
            .map(flatten)
            .map(self.quantize)
            .map(self.shuffle_and_add_indices)
            .cache()
        )
        dataset_train_noshuf = (
            self.ds_train_orig
            .map(normalize_image)
            .map(flatten)
            .map(self.quantize)
            .map(self.add_indices)
            .cache()
        )
        
        dataset_train = (
            dataset_train_shuf
            .repeat()
            .shuffle(self.config['dataset']['buffer_size'])
        )
            
        if self.config.batch_size_schedule == 'dynamic':
            dataset_train = dataset_train.batch(self.config.minibatch_size)
            dataset_train = dataset_train.map(self.add_n_from_distribution)
            dataset_train = dataset_train.batch(self.config.end_accum_steps * self.config.num_devices)
        else:
            dataset_train = dataset_train.batch(self.config.global_batch_size)
            dataset_train = dataset_train.map(self.add_n_from_distribution)
        
        dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)
            
        
        dataset_test = (
            self.ds_test_orig
            .map(normalize_image)
            .map(flatten)
            .map(self.quantize)
            .map(self.add_indices)
            .cache()
            .repeat()
            .shuffle(self.config['dataset']['buffer_size'])
            .batch(self.config['test_minibatch_size'], drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        dataset_test_shuffled = (
            self.ds_test_orig
            .map(normalize_image)
            .map(flatten)
            .map(self.quantize)
            .map(self.shuffle_and_add_indices)
            .cache()
            .repeat()
            .shuffle(self.config['dataset']['buffer_size'])
            .batch(self.config['test_minibatch_size'], drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        return dataset_train, dataset_test, dataset_test_shuffled