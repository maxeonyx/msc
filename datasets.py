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
from icecream import ic
ic.configureOutput(includeContext=True)

def ignore_label(image, label):
    return image

def is_right_hand(filename, n_frames, angles, is_right_hand):
    return is_right_hand == True

def ignore_metadata(filename, n_frames, angles, is_right_hand):
    return angles

def normalize_image(image):
    return tf.cast(image, dtype=tf.float32) / 255.0

def find_centroids(config, ds_train):
    num_clusters = config.dataset.n_colors
    n_color_dims = config.dataset.n_color_dims
    batch_size = config.kmeans_batch_size
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=batch_size, verbose=False)
    ds_batched = ds_train.map(normalize_image)
    data = next(iter(ds_batched))
    pixels = tf.reshape(data, [-1, n_color_dims])
    kmeans.fit(pixels)
#     with enlighten.get_manager() as manager:
#         title = manager.status_bar(f"K-Means clustering to make {num_clusters}-color MNIST Dataset", justify=enlighten.Justify.CENTER)
#         clusters_names = manager.status_bar(''.join('{:<10}'.format(f"cen. {i}") for i in range(num_clusters)))
#         clusters_status = manager.status_bar(''.join('{:<10}'.format('??????') for _ in range(num_clusters)))
#         pbar = manager.counter(total=60000//batch_size, desc=f'Discretize to {num_clusters} colors', unit='minibatches')
#         for img, _ in pbar(iter(ds_batched)):
#             pixels = tf.reshape(img, [-1, n_color_dims]).numpy()
#             kmeans.partial_fit(pixels)
            
#             clusters_status.update(''.join('[' + ', '.join('{:<0.1f}'.format(x[i]) for i in range(len(x))) + ']' for x in np.sort(kmeans.cluster_centers_, axis=0)))

    centroids = kmeans.cluster_centers_
    centroids = tf.convert_to_tensor(np.sort(centroids, axis=0), dtype=tf.float32)
    centroids = tf.reshape(centroids, [num_clusters, n_color_dims])
    return centroids

def mnist_gamma_distribution():
    alpha, beta = 1.2, 0.007
    gamma = tfp.distributions.Gamma(concentration=alpha, rate=beta)
    return gamma, f"Gamma (alpha={alpha}, beta={beta})"

def gamma_distribution_7x7():
    alpha, beta = 1.5, 0.1
    gamma = tfp.distributions.Gamma(concentration=alpha, rate=beta)
    return gamma, f"Gamma (alpha={alpha}, beta={beta})"

def gamma_distribution_12x12():
    alpha, beta = 1.3, 0.03
    gamma = tfp.distributions.Gamma(concentration=alpha, rate=beta)
    return gamma, f"Gamma (alpha={alpha}, beta={beta})"
    
def plot_distribution(config, dist, name):

    def plot_cdf():
        x = tf.range(config.dataset.seq_length, dtype=tf.float32)
        cdf = dist.cdf(x)
        fig, ax = plt.subplots()
        ax.plot(tf.range(config.dataset.seq_length), cdf)
        ax.set_title(f"{name} CDF, distribution of examples by number of pixels.")


    def plot_pdf():
        x = tf.range(config.dataset.seq_length, dtype=tf.float32)
        pdf = dist.prob(x)
        fig, ax = plt.subplots()
        ax.plot(tf.range(config.dataset.seq_length), pdf)
        ax.set_title(f"{name} PDF, distribution of examples by number of pixels.")


    def test_sample():

        return tf.cast(tf.math.minimum(tf.math.round(dist.sample(sample_shape=[100])), config.dataset.seq_length - 1), tf.int32)

    plot_pdf()
    plot_cdf()
    print("Example integers sampled from gamma distribution.")
    print(test_sample())

def squared_distance(a, b):
    """
    given two tensors a and b, return the distance between the final dim of a and each b
    
    input shape a [..., A, D]
    input shape b [B, D]
    
    output shape [..., A, B]
    
    """
    a = tf.expand_dims(a, -2)
    b = tf.expand_dims(b, 0)
    return tf.math.reduce_sum(tf.math.squared_difference(a, b), axis=-1)

def flatten(image):
    shape = tf.shape(image) # (height, width, color)
    sequence = tf.reshape(image, (-1, shape[2])) # (height * width, color)
    return sequence

def flatten_hands(data):
    shape = tf.shape(data) # (n_frames, n_dof)
    sequence = tf.reshape(data, [-1]) # (height * width, color)
    return sequence

class Datasets:
    
    def __init__(self, config, ds_train_orig, ds_test_orig, centroids, distribution):
        
        self.centroids = centroids
        self.config = config
        self.dist = distribution
        self.ds_train_orig = ds_train_orig
        self.ds_test_orig = ds_test_orig
    
    def quantize(self, sequence):
        print("before quant", sequence.shape)
        d = squared_distance(sequence, self.centroids) # (height * width, n_centroids)
        sequence = tf.math.argmin(d, axis=1, output_type=tf.int32)  # (height * width)
        print("after  quant", sequence.shape)
        return sequence

    def unquantize(self, x, to="same"):
        if x.shape[-1] == 1:
            x = tf.squeeze(x, -1)
        x_one_hot = tf.cast(tf.one_hot(x, depth=len(self.centroids)), dtype=tf.float32)  # (seq, num_centroids)
        if to == "rgb":
            if self.centroids.shape[-1] == 1:
                centroids = tf.tile(self.centroids, [1, 3])
            elif self.centroids.shape[-1] == 3:
                centroids = self.centroids
            else:
                assert False, "color_dims is not 1 or 3, can't use to='rgb'"
        else:
            centroids = self.centroids
        y = tf.linalg.matmul(x_one_hot,centroids)  # (seq, num_features, n_color_dims)
        return y
    
    def flatten_color_dim(self, sequence):
        new_shape = [*sequence.shape[:-2], sequence.shape[-2] * sequence.shape[-1]]
        sequence = tf.reshape(sequence, new_shape)
        return sequence
    
    def reinvent_color_dim(self, sequence):
        new_shape = [*sequence.shape[:-1], sequence.shape[-1]//self.config.dataset.n_color_dims, self.config.dataset.n_color_dims]
        sequence = tf.reshape(sequence, new_shape)
        return sequence
    
    def reinvent_hand_dim(self, sequence):
        assert sequence.shape[-1]//self.config.dataset.n_dof == self.config.dataset.n_frames, f"sequence.shape[-1]//self.config.dataset.n_dof == {sequence.shape[-1]//self.config.dataset.n_dof} != {self.config.dataset.n_frames} == self.config.dataset.n_frames"
        new_shape = [*sequence.shape[:-1], sequence.shape[-1]//self.config.dataset.n_dof, self.config.dataset.n_dof]
        sequence = tf.reshape(sequence, new_shape)
        return sequence
   
    
    def to_grayscale_rgb(self, seq):
        assert seq.shape[-1] == 1, "to_grayscale() only works if the final dim is length 1"
        # turn the final dim from [1] to [3] by filling the color value to r, g, and b
        seq = tf.tile(seq, [*[1 for _ in seq.shape[:-1]], 3])
        return seq

    def expected_col(self, dist_params):

        if self.config.dataset.discrete:

            probs = tf.nn.softmax(dist_params, axis=2)

            # shape: (batch, seq, n_colors)
            
            probs = tf.expand_dims(probs, axis=-1)
            # shape: (batch, seq, n_colors, n_color_dim=1)
            
            # add batch and sequence dim to centroids
            centroids = tf.expand_dims(tf.expand_dims(self.centroids, axis=0), axis=0)
            # shape: (batch=1, seq=1, n_colors, n_color_dim)
            
            # this does linear interpolation between colors
            expected_color = probs * centroids
            # shape: (batch=1, seq=1, n_colors, n_color_dim)
            expected_color = tf.reduce_sum(expected_color, axis=[-2])
            # shape: (batch=1, seq=1, n_color_dim)
            
            return expected_color
        else:
            if self.config.dataset.loss == 'gaussian':
                return dist_params[:, :, 0, None] # means (incl. color dim)
    
    def shuffle_and_add_indices(self, sequence):

        idxs = tf.range(self.config.dataset.seq_length, dtype=tf.int32)
        shuf_idxs = tf.random.shuffle(idxs)

        shuf_sequence = tf.gather(sequence, shuf_idxs)

        return sequence, idxs, shuf_sequence, shuf_idxs
    
    def add_noise(self, sequence, idxs, shuf_sequence, shuf_idxs):
        if self.config.dataset.noise_fraction:
            n_noise = int(self.config.dataset.noise_fraction * self.config.dataset.seq_length)
            if self.config.dataset.continuous:
                noise = tf.random.uniform([n_noise], minval=0., maxval=1., dtype=tf.float32)
            else:
                noise = tf.random.uniform([n_noise], minval=0, maxval=self.config.dataset.n_colors, dtype=tf.int32)
            rand_idxs = tf.random.shuffle(idxs)
            rand_idxs = rand_idxs[:n_noise, None]
            shuf_sequence_noise = tf.tensor_scatter_nd_update(shuf_sequence, rand_idxs, noise)
        else:
            shuf_sequence_noise = shuf_sequence
        
        return sequence, idxs, shuf_sequence, shuf_idxs, shuf_sequence_noise

    def add_n_from_distribution(self, sequence, *inputs):
        bs = tf.shape(sequence)[0]
        # sample single integer between 0 and 783 from given distribution
        n = tf.cast(tf.math.round(self.dist.sample(sample_shape=[])), tf.int32)
        n = tf.clip_by_value(n, 1, self.config.dataset.seq_length - 1)
        
        if self.config.grad_accum_steps is not None:
            return sequence, *inputs, n
        else:
            return sequence, *inputs, tf.repeat(n, self.config.num_devices)

    def rescale(self, image):
        image = tf.image.resize(image, self.config.dataset.rescale)
        return image
    
    def deg2rad(self, angles):
        angles = (angles / 360. ) * 2. * np.pi
        return angles

    def circular_mean(self, angles):
        # compute the circular mean of the data for this example+track
        # rotate the data so that the circular mean is 0
        # store the circular mean
        means_cos_a = tf.reduce_mean(tf.math.cos(angles), axis=0)
        means_sin_a = tf.reduce_mean(tf.math.sin(angles), axis=0)
        circular_means = tf.math.atan2(means_sin_a, means_cos_a)
        return circular_means

    def recluster(self, angles):
        # rotate the data so the circular mean is 0
        circular_means = self.circular_mean(angles)
        angles = angles - circular_means[None, :]
        angles = tf.cond(angles < np.pi, lambda: angles+np.pi*2, lambda: angles)
        return angles

    def chunk_flatten_add_multiidxs(self, angles):
        n_total_frames = tf.shape(angles)[0]
        
        n_dof = tf.shape(angles)[1]
        n_chunk_frames = self.config.dataset.n_frames
        i = tf.random.uniform(shape=[], minval=0, maxval=n_total_frames*n_dof-n_chunk_frames*n_dof, dtype=tf.int32)
        # angles = tf.reshape(angles, [-1])[i:i+n_chunk_frames*n_dof]
        # idxs = tf.range(i%n_dof, i%n_dof+n_chunk_frames*n_dof)
        
        frame_i = tf.random.uniform(shape=[], minval=0, maxval=n_total_frames-n_chunk_frames, dtype=tf.int32)
        frame_idxs = tf.range(frame_i, frame_i+n_chunk_frames)
        frame_idxs = tf.reshape(frame_idxs, [n_chunk_frames, 1])
        frame_idxs = tf.tile(frame_idxs, [1, n_dof])

        dof_idxs = tf.range(n_dof)
        dof_idxs = tf.reshape(dof_idxs, [1, n_dof])
        dof_idxs = tf.tile(dof_idxs, [n_chunk_frames, 1])
        
        return angles, frame_idxs, dof_idxs

    def chunk_flatten_add_idxs(self, angles):
        n_total_frames = tf.shape(angles)[0]
        
        n_dof = tf.shape(angles)[1]
        n_chunk_frames = self.config.dataset.n_frames
        frame_i = tf.random.uniform(shape=[], minval=0, maxval=n_total_frames-n_chunk_frames, dtype=tf.int32)
        i = frame_i*n_dof
        angles = tf.reshape(angles, [-1])[i:i+n_chunk_frames*n_dof]
        idxs = tf.range(i, i+n_chunk_frames*n_dof, dtype=tf.int32)
        return angles, idxs
        
    def make_datasets(self, for_statistics=False):
        
        dataset_train = self.ds_train_orig
        dataset_test = self.ds_test_orig
        
        if self.config.dataset.type == 'image':
            dataset_train = dataset_train.map(normalize_image)
            dataset_test = dataset_test.map(normalize_image)

            if self.config.dataset.rescale:
                dataset_train = dataset_train.map(self.rescale)
                dataset_test = dataset_test.map(self.rescale)
        
            dataset_train = dataset_train.map(flatten)
            dataset_test = dataset_test.map(flatten)
            if self.config.dataset.continuous:
                dataset_test = dataset_test.map(self.flatten_color_dim)
                dataset_train = dataset_train.map(self.flatten_color_dim)
            else:
                dataset_test = dataset_test.map(self.quantize)
                dataset_train = dataset_train.map(self.quantize)

        if self.config.dataset.type == 'hands':
            dataset_train = (
                dataset_train
                .map(self.deg2rad)
                .map(self.recluster)
            )
            dataset_test = (
                dataset_test
                .map(self.deg2rad)
                .map(self.recluster)
            )
            ic()
        
        dataset_test = dataset_test.cache()
        dataset_train = dataset_train.cache()
        
        if for_statistics:
            train = next(iter(dataset_train.batch(60000)))
            test = next(iter(dataset_test.batch(10000)))
            return train, test
        
        dataset_test = dataset_test.repeat()
        dataset_train = dataset_train.repeat()

        if self.config.dataset.type == 'hands':
            dataset_train = (
                dataset_train
                .map(self.chunk_flatten_add_idxs)
            )
            dataset_test = (
                dataset_test
                .map(self.chunk_flatten_add_idxs)
            )
            ic()
        
        if self.config.dataset.type == 'image':
            dataset_test = (
                dataset_test
                .map(self.shuffle_and_add_indices)
            )
            
            dataset_train = (
                dataset_train
                .map(self.shuffle_and_add_indices)
            )

        dataset_test = (
            dataset_test
            .shuffle(self.config.dataset.buffer_size)
            .batch(self.config.test_minibatch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        
        dataset_train = (
            dataset_train
            .shuffle(self.config.dataset.buffer_size)
        )
        
        if self.config.dataset.noise_fraction:
            dataset_train = dataset_train.map(self.add_noise)
        
        
        def n_split(d):
            if self.dist:
                return d.map(self.add_n_from_distribution)
            else:
                return d
        
        if self.config.grad_accum_steps is None or self.config.grad_accum_steps == 1:
            dataset_train = dataset_train.batch(self.config.global_batch_size)
            dataset_train = n_split(dataset_train)
            print("Not using gradient accumulation")
        else:
            dataset_train = dataset_train.batch(self.config.minibatch_size)
            dataset_train = n_split(dataset_train)
            dataset_train = dataset_train.batch(self.config.max_accum_steps * self.config.num_devices)
            print("Using gradient accumulation")
            
        dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset_train, dataset_test
