"""
Takes a np BVH dataset and returns a model-agnostic tf.data.Dataset pipeline, which can be further transformed.
"""

import sys
from math import pi, tau

import numpy as np
import tensorflow as tf
import einops as ein
from einops import rearrange, reduce, repeat

from ml import utils

from . import data_bvh


def add_idx_arrays(x, frame_idxs_only=False):
    """
    Add idx arrays to the dataset
    """
    shape = tf.shape(x["angles"])
    return x | {
        "frame_idxs": tf.tile(tf.range(shape[0], dtype=tf.int32)[:, None, None], (1, shape[1], shape[2])),
        "hand_idxs": tf.tile(tf.range(shape[1], dtype=tf.int32)[None, :, None], (shape[0], 1, shape[2])),
        "dof_idxs": tf.tile(tf.range(shape[2], dtype=tf.int32)[None, None, :], (shape[0], shape[1], 1)),
    }


# flatten the angle and *_idx tensors
def flatten(x):
    
    x["angles"] = tf.reshape(x["angles"], [-1])
    x["frame_idxs"] = tf.reshape(x["frame_idxs"], [-1])
    x["hand_idxs"] = tf.reshape(x["hand_idxs"], [-1])
    x["dof_idxs"] = tf.reshape(x["dof_idxs"], [-1])

    return x


def frame_idxs_for(cfg, offset_in_frame, seq_length):
    """
    frame idxs are reversed and relative

    offset_in_frame should be something like:
       hand_idx * cfg.n_dof + dof_idx

    returns a tensor of shape [seq_length]
    eg. with offset_in_frame = 1, and seq_length = 9

    tokens:     [ A2, A3, B1, B2, B3, C1, C2, C3, D1]
    frame idxs: [ 3,  3,  2,  2,  2,  1,  1,  1,  0 ]
    
    the first and last frames may not be complete
    """
    tok_per_frame = cfg.n_hands * cfg.n_dof
    
    # make a list of frame idxs that the sequence contains
    n_frames = (seq_length // tok_per_frame) + 2
    frame_idxs = tf.reverse(tf.range(n_frames * tok_per_frame) // tok_per_frame, axis=[0])
    n_frame_idxs = tf.shape(frame_idxs)[0]
    return frame_idxs[n_frame_idxs-tok_per_frame+offset_in_frame-seq_length:n_frame_idxs-tok_per_frame+offset_in_frame]


# cut a fixed size chunk from the tensor at a random token index
def random_flat_chunk(cfg, size, x, aligned=False):
    shape = tf.shape(x["angles"])
    n_frames = shape[0]
    
    x["angles"] = rearrange(x["angles"], 'b, f, h, d -> b, f, (h, d)')
    x["frame_idxs"] = rearrange(x["frame_idxs"], 'b, f, h, d -> b, f, (h, d)')
    x["hand_idxs"] = rearrange(x["hand_idxs"], 'b, f, h, d -> b, f, (h, d)')
    x["dof_idxs"] = rearrange(x["dof_idxs"], 'b, f, h, d -> b, f, (h, d)')

    tok_per_frame = cfg.n_hands * cfg.n_dof
    chunk_size = size
    chunk_toks = chunk_size * tok_per_frame

    if aligned:
        # get a random index
        idx = tf.random.uniform([], minval=0, maxval=n_frames-chunk_size, dtype=tf.int32)
        idx = idx * tok_per_frame
    else:
        idx = tf.random.uniform([], minval=0, maxval=n_frames*tok_per_frame-chunk_toks, dtype=tf.int32)

    x["angles"] = x["angles"][idx:idx+chunk_toks]

    if cfg.relative_frame_idxs:
        x["frame_idxs"] = frame_idxs_for(cfg, idx % tok_per_frame, chunk_toks)
    else:
        x["frame_idxs"] = x["frame_idxs"][idx:idx+chunk_toks]


    x["hand_idxs"] = x["hand_idxs"][idx:idx+chunk_toks]
    x["dof_idxs"] = x["dof_idxs"][idx:idx+chunk_toks]

    return x

# def random_foveal_chunk(cfg, chunk_size, x):

#     idx = tf.random.uniform([], minval=0, maxval=n_frames-chunk_size, dtype=tf.int32)
#     idx = idx * tok_per_frame


def to_train_input_and_target(cfg, x):
    inp = x.copy()

    inp["angles"] = inp["angles"][..., :-1]
    inp["frame_idxs"] = inp["frame_idxs"][..., :-1]
    inp["hand_idxs"] = inp["hand_idxs"][..., :-1]
    inp["dof_idxs"] = inp["dof_idxs"][..., :-1]

    inp_shape = [cfg.batch_size, cfg.chunk_size * cfg.n_hands * cfg.n_dof - 1]
    inp["angles"] = tf.ensure_shape(inp["angles"], inp_shape)
    inp["frame_idxs"] = tf.ensure_shape(inp["frame_idxs"], inp_shape)
    inp["hand_idxs"] = tf.ensure_shape(inp["hand_idxs"], inp_shape)
    inp["dof_idxs"] = tf.ensure_shape(inp["dof_idxs"], inp_shape)

    tar = x["angles"]
    tar_shape = [cfg.batch_size, cfg.chunk_size * cfg.n_hands * cfg.n_dof]
    tar = tf.ensure_shape(tar, tar_shape)

    return (inp, tar)


def to_test_input_and_target(cfg, x):
    inp = x.copy()

    seq_len = tf.shape(inp["angles"])[-1]

    predict_tokens = cfg.predict_frames * cfg.n_hands * cfg.n_dof

    inp["angles"] = inp["angles"][..., :seq_len-predict_tokens]
    inp["frame_idxs"] = inp["frame_idxs"][..., :seq_len-predict_tokens]
    inp["hand_idxs"] = inp["hand_idxs"][..., :seq_len-predict_tokens]
    inp["dof_idxs"] = inp["dof_idxs"][..., :seq_len-predict_tokens]

    tar = x["angles"]

    return (inp, tar)


def subset(cfg, angles):
    """
    Slice just some of the data.
    """
    angles = angles[:, :cfg.n_hands, :cfg.n_dof]

    return angles


def recluster(angles, circular_means):
    angles = utils.recluster(angles, frame_axis=0, circular_means=circular_means)
    return angles


def make_decimate_fn(cfg, n_hands, n_dof):
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, n_hands, n_dof], dtype=tf.float32),
    ])
    def decimate(angles):
        angles = rearrange(angles, 'f h d -> f (h d)')
        len_angles = tf.shape(angles)[0]
        new_angles = angles[:1, :]
        for i in tf.range(1, len_angles):
            if tf.linalg.norm(angles[i] - new_angles[-1]) > cfg.decimate_threshold:
                new_angles = tf.concat([new_angles, angles[i:i+1]], axis=0)
        new_angles = rearrange(new_angles, 'f (h d) -> f h d', h=n_hands, d=n_dof)
        return new_angles

    return decimate

def random_flat_vector_chunk(cfg, x):
    """
    Take a random flat chunk of the dataset.
    """

    n_frames = x["angles"].shape[0]
    i = tf.random.uniform(shape=(), minval=0, maxval=n_frames-cfg.chunk_size, dtype=tf.int32)

    x["angles"] = x["angles"][i:i+cfg.chunk_size]

    return x


def synthetic_data(cfg, seed=1234):
    """
    Create a synthetic dataset compatible with the BVH one.
    """

    tf.random.set_seed(seed)

    n_examples = cfg.get("n_examples", 65)

    n_frames = tf.random.uniform(shape=[n_examples], minval=4000, maxval=8000, dtype=tf.int32, seed=seed)

    def make_angle_track(length):
        """
        Makes a track with random movement. One degree of freedom.
        """
        i = tf.range(0, length, dtype=tf.float32)
        sin_freqs = tf.exp(tf.random.uniform(shape=[3], minval=-2, maxval=0, dtype=tf.float32, seed=seed))
        sin_offsets = tf.random.uniform(shape=[3], minval=0, maxval=2*np.pi, dtype=tf.float32, seed=seed)
        sin_amplitudes = tf.random.uniform(shape=[3], minval=tau/48, maxval=tau/6, dtype=tf.float32, seed=seed)
        mean = tf.random.uniform(shape=[], minval=-pi, maxval=pi, dtype=tf.float32, seed=seed)
        angles = tf.reduce_sum(tf.sin(sin_freqs[None, :] * i[:, None] + sin_offsets[None, :])*sin_amplitudes[None, :], axis=1)
        angles = utils.angle_wrap(angles + mean)

        return angles

    angles = tf.concat([
        tf.stack([
            make_angle_track(length=n_frames[i_example])
            for i_track in range(cfg.n_hands * cfg.n_joints_per_hand * cfg.n_dof_per_joint)
        ], axis=1)
        for i_example in range(n_examples)
    ], axis=0)

    angles = ein.rearrange(angles, 'f (h j d) -> f h (j d)', h=cfg.n_hands, j=cfg.n_joints_per_hand, d=cfg.n_dof_per_joint)

    angles = tf.RaggedTensor.from_row_lengths(angles, n_frames)

    dataset = tf.data.Dataset.from_tensor_slices(angles)
    
    return dataset

def bvh_data(cfg):
    
    _filenames, angles, n_frames = data_bvh.np_dataset_parallel_lists(force=cfg.force, columns=cfg.columns)
    
    all_angles = tf.concat(angles, axis=0)
    n_frames = tf.constant(n_frames)
    orig_n_hands = all_angles.shape[1]
    orig_n_dof = all_angles.shape[2]
    ragged_angles = tf.RaggedTensor.from_row_lengths(all_angles, n_frames)

    dataset = dataset.map(lambda x: subset(cfg, x))

    if cfg.recluster:
        circular_means = utils.circular_mean(all_angles, axis=0)
        dataset = dataset.map(lambda x: recluster(x, circular_means))
    
    if cfg.decimate:
        decimate = make_decimate_fn(cfg, orig_n_hands, orig_n_dof)
        dataset = dataset.map(decimate)

    dataset = tf.data.Dataset.from_tensor_slices(ragged_angles)

    return dataset

# take np BVH dataset and return angles, hand idxs, frame idxs, and dof idxs
def tf_dataset(cfg, finish_fn, data_fn=bvh_data):
    """
    Takes a np BVH dataset, of type []
    returns a model-agnostic tf.data.Dataset pipeline, which can be further transformed.

    returns a tf Dataset with element tuple (filename, angles, hand_idxs, frame_idxs, dof_idxs),
    each with shape (n_hands, n_frames, n_dof)
    """

    dataset = data_fn(cfg)

    dataset = dataset.map(lambda a: { "angles": a })
    
    dataset = dataset.map(add_idx_arrays)

    # dataset = dataset.snapshot(cfg.cached_dataset_path, compression=None)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=dataset.cardinality(), seed=1234) # keep seed the same for reproducibility of test error

    test_dataset = (
        dataset
        .take(12) # take 12 (~10%) to test
        .repeat()
    )
    val_dataset = (
        dataset
        .skip(12)
        .take(12) # take the next 12 (~10%) to validation
        .repeat()
    )
    train_dataset = (
        dataset
        .skip(24)
        .repeat()
        .shuffle(buffer_size=cfg.shuffle_buffer_size)
    )
    
    train_dataset, test_dataset, val_dataset = finish_fn(cfg, train_dataset, test_dataset, val_dataset)

    train_dataset = (
        train_dataset
        .take(cfg.steps)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    test_dataset = (
        test_dataset
        .take(cfg.test_steps)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    val_dataset = (
        val_dataset
        .take(cfg.test_steps)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return train_dataset, test_dataset, val_dataset

def pre_dataset(cfg, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset):
    
    # train input isn't always frame aligned
    train_dataset = train_dataset.map(lambda x: random_flat_chunk(cfg, cfg.chunk_size, x))
    train_dataset = train_dataset.batch(cfg.batch_size)
    # split into input and target, with 1 token offset
    train_dataset = train_dataset.map(lambda x: to_train_input_and_target(cfg, x))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # test input is always frame-aligned
    # take fixed size chunks from the tensor at random frame indices
    test_dataset = test_dataset.repeat(100) # N = 100 repeats * 12 examples = 1200 chunks
    test_dataset = test_dataset.map(lambda x: random_flat_chunk(cfg, cfg.chunk_size + cfg.predict_frames, x, aligned=True, seed=1234))
    test_dataset = test_dataset.map(lambda x: to_test_input_and_target(cfg, x))
    # test dataset doesn't need to be split
    test_dataset = test_dataset.batch(cfg.test_batch_size)

    return train_dataset, test_dataset
