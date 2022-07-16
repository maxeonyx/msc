"""
Takes a np BVH dataset and returns a model-agnostic tf.data.Dataset pipeline, which can be further transformed.
"""

import sys
import math

import numpy as np
import tensorflow as tf
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
    
    x["angles"] = tf.reshape(x["angles"], [-1])
    x["frame_idxs"] = tf.reshape(x["frame_idxs"], [-1])
    x["hand_idxs"] = tf.reshape(x["hand_idxs"], [-1])
    x["dof_idxs"] = tf.reshape(x["dof_idxs"], [-1])
    
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


def subset(cfg, x):
    """
    Slice just some of the data.
    """
    x["angles"] = x["angles"][:, :cfg.n_hands, :cfg.n_dof]

    return x


def recluster(x, circular_means):
    x["angles"] = utils.recluster(x["angles"], frame_axis=0, circular_means=circular_means)
    return x


def make_decimate_fn(cfg):
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, cfg.n_hands, cfg.n_dof], dtype=tf.float32),
    ])
    def decimate(angles):
        angles = rearrange(angles, 'f h d -> f (h d)')
        new_angles = angles[:1, :]
        for i in tf.range(1, angles.shape[0]):
            if tf.linalg.norm(angles[i] - new_angles[-1]) > cfg.decimate_threshold:
                new_angles = tf.concat([new_angles, angles[i:i+1]], axis=0)
        angles = rearrange(angles, 'f (h d) -> f h d', h=cfg.n_hands, d=cfg.n_dof)
        return angles
    
    def decimate_map(x):
        x["angles"] = decimate(x["angles"])
        return x

    return decimate, decimate_map


def to_dict(filename, angles):
    return {
        # "filename": filename,
        "angles": angles,
    }


def add_frame_idxs(x):
    x["frame_idxs"] = tf.range(tf.shape(x["angles"])[0])
    return x


def to_sin_cos(x):
    sin = tf.sin(x["angles"])
    cos = tf.cos(x["angles"])
    x["angles"] = rearrange([sin, cos], 'sincos ... -> ... sincos')
    return x


def random_flat_vector_chunk(cfg, x):
    """
    Take a random flat chunk of the dataset.
    """

    n_frames = x["angles"].shape[0]
    i = tf.random.uniform(shape=(), minval=0, maxval=n_frames-cfg.chunk_size, dtype=tf.int32)

    x["angles"] = x["angles"][i:i+cfg.chunk_size]

    return x


# take np BVH dataset and return angles, hand idxs, frame idxs, and dof idxs
def tf_dataset(cfg):
    """
    Takes a np BVH dataset, of type []
    returns a model-agnostic tf.data.Dataset pipeline, which can be further transformed.

    returns a tf Dataset with element tuple (filename, angles, hand_idxs, frame_idxs, dof_idxs),
    each with shape (n_hands, n_frames, n_dof)
    """

    filenames, angles, n_frames = data_bvh.np_dataset_parallel_lists(cfg.force)
    filenames = tf.constant(filenames)
    all_angles = tf.concat(angles, axis=0)
    n_frames = tf.constant(n_frames)
    ragged_angles = tf.RaggedTensor.from_row_lengths(all_angles, n_frames)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, ragged_angles))

    dataset = dataset.map(to_dict)
    dataset = dataset.map(lambda x: subset(cfg, x))

    if cfg.recluster:
        circular_means = utils.circular_mean(all_angles, axis=0)[:cfg.n_hands, :cfg.n_dof]
        dataset = dataset.map(lambda x: recluster(x, circular_means))
    
    if cfg.decimate:
        _, decimate = make_decimate_fn(cfg)
        dataset = dataset.map(decimate)
    
    dataset = dataset.map(add_idx_arrays)

    dataset = dataset.snapshot(cfg.cached_dataset_path, compression=None)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=cfg.shuffle_buffer_size)
    
    if cfg.vector:
        return tf_dataset_vector(cfg, dataset)
    else:
        return tf_dataset_scalar(cfg, dataset)

def tf_dataset_scalar(cfg, dataset):
    
    # train input isn't always frame aligned
    train_dataset = dataset.map(lambda x: random_flat_chunk(cfg, cfg.chunk_size, x))
    train_dataset = train_dataset.batch(cfg.batch_size)
    # split into input and target, with 1 token offset
    train_dataset = train_dataset.map(lambda x: to_train_input_and_target(cfg, x))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # test input is always frame-aligned
    # take fixed size chunks from the tensor at random frame indices
    test_dataset = dataset.map(lambda x: random_flat_chunk(cfg, cfg.chunk_size + cfg.predict_frames, x, aligned=True))
    test_dataset = test_dataset.map(lambda x: to_test_input_and_target(cfg, x))
    # test dataset doesn't need to be split
    test_dataset = test_dataset.batch(cfg.test_batch_size)

    return train_dataset, test_dataset
    

def tf_dataset_vector(cfg, dataset):
    
    # ignore hand_idxs and dof_idxs
    dataset = dataset.map(lambda x: {"angles": x["angles"], "frame_idxs": x["frame_idxs"]})

    dataset = dataset.map(lambda x: random_flat_vector_chunk(cfg, x))
    dataset = dataset.map(to_sin_cos)

    train_dataset = dataset.batch(cfg.batch_size)
    # split into input and target, with 1 token offset
    train_dataset = train_dataset.map(lambda x: to_train_input_and_target(cfg, x))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # test input is always frame-aligned
    test_dataset = test_dataset.map(lambda x: to_test_input_and_target(cfg, x))
    # test dataset doesn't need to be split
    test_dataset = test_dataset.batch(1)

    return train_dataset, test_dataset
