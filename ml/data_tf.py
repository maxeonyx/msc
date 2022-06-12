"""
Takes a np BVH dataset and returns a model-agnostic tf.data.Dataset pipeline, which can be further transformed.
"""

import sys
from xml.dom import NotFoundErr

import numpy as np
import tensorflow as tf

from . import data_bvh


def add_idx_arrays(filename, angles):
    """
    Add idx arrays to the dataset
    """
    shape = tf.shape(angles)
    return {
        "filename": filename,
        "angles": angles,
        "frame_idxs": tf.tile(tf.range(shape[0])[:, None, None], (1, shape[1], shape[2])),
        "hand_idxs": tf.tile(tf.range(shape[1])[None, :, None], (shape[0], 1, shape[2])),
        "dof_idxs": tf.tile(tf.range(shape[2])[None, None, :], (shape[0], shape[1], 1)),
    }

# cut a fixed size chunk from the tensor at a random frame index
def random_chunk(cfg, x):
    shape = tf.shape(x["angles"])
    n_frames = shape[0]
    # get a random index
    idx = tf.random.uniform([], minval=0, maxval=n_frames-cfg.chunk_size, dtype=tf.int32)

    x["angles"] = x["angles"][idx:idx+cfg.chunk_size, :, :]
    x["frame_idxs"] = x["frame_idxs"][idx:idx+cfg.chunk_size, :, :]
    x["hand_idxs"] = x["hand_idxs"][idx:idx+cfg.chunk_size, :, :]
    x["dof_idxs"] = x["dof_idxs"][idx:idx+cfg.chunk_size, :, :]

    return x

# flatten the angle and *_idx tensors
def flatten(x):
    x = x.copy()
    x["angles"] = tf.reshape(x["angles"], [-1])
    x["frame_idxs"] = tf.reshape(x["frame_idxs"], [-1])
    x["hand_idxs"] = tf.reshape(x["hand_idxs"], [-1])
    x["dof_idxs"] = tf.reshape(x["dof_idxs"], [-1])
    return x

def to_input(cfg, old_x):
    x = old_x.copy()

    # add "begin" sentinel value to the start of sequences
    x["angles"] = tf.concat([tf.constant(999, dtype=tf.float32)[None], x["angles"][:-1]], axis=-1)
    x["frame_idxs"] = tf.concat([tf.constant(cfg.chunk_size)[None], x["frame_idxs"][:-1]], axis=-1)
    x["hand_idxs"] = tf.concat([tf.constant(cfg.n_hands)[None], x["hand_idxs"][:-1]], axis=-1)
    x["dof_idxs"] = tf.concat([tf.constant(cfg.n_dof)[None], x["dof_idxs"][:-1]], axis=-1)

    return x

def to_input_and_target(cfg, old_x):
    x = old_x.copy()

    # add "begin" sentinel value to the start of sequences
    x["angles"] = tf.concat([tf.constant(999, dtype=tf.float32)[None], x["angles"][:-1]], axis=-1)
    x["frame_idxs"] = tf.concat([tf.constant(cfg.chunk_size)[None], x["frame_idxs"][:-1]], axis=-1)
    x["hand_idxs"] = tf.concat([tf.constant(cfg.n_hands)[None], x["hand_idxs"][:-1]], axis=-1)
    x["dof_idxs"] = tf.concat([tf.constant(cfg.n_dof)[None], x["dof_idxs"][:-1]], axis=-1)

    target = old_x["angles"]

    tf.ensure_shape(target, [cfg.chunk_size * cfg.n_hands * cfg.n_dof])

    return (x, target)


# take np BVH dataset and return angles, hand idxs, frame idxs, and dof idxs
def tf_dataset(cfg):
    """
    Takes a np BVH dataset, of type []
    returns a model-agnostic tf.data.Dataset pipeline, which can be further transformed.

    returns a tf Dataset with element tuple (filename, angles, hand_idxs, frame_idxs, dof_idxs),
    each with shape (n_hands, n_frames, n_dof)
    """

    if cfg.force:
        print(f"Forcing creation of {cfg.cached_dataset_path}", file=sys.stderr)
        create_dataset = True
    else:
        try:
            dataset = tf.data.experimental.load(cfg.cached_dataset_path)
            create_dataset = False
        except tf.errors.NotFoundError as f:
            print(f"{cfg.cached_dataset_path} not found, creating...", file=sys.stderr)
            create_dataset = True
    
    if create_dataset:
        dataset = tf.data.Dataset.from_generator(
            lambda: data_bvh.get_bvh_data(convert_deg_to_rad=cfg.convert_deg_to_rad),
            output_signature=(
                tf.TensorSpec(name="filename", shape=(), dtype=tf.string),
                tf.TensorSpec(name="angles", shape=[None, cfg.n_hands, cfg.n_dof], dtype=tf.float32),
            )
        )

        dataset = dataset.map(add_idx_arrays)

        tf.data.experimental.save(dataset, cfg.cached_dataset_path)

    dataset = dataset.cache()

    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=cfg.shuffle_buffer_size)
    
    # take fixed size chunks from the tensor at random frame indices
    dataset = dataset.map(lambda x: random_chunk(cfg, x))
    # flatten 3-D tensors to 1-D sequences
    dataset = dataset.map(flatten)

    # split into input and target, with 1 token offset
    train_dataset = dataset.map(lambda x: to_input_and_target(cfg, x))
    train_dataset = train_dataset.batch(cfg.batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # test dataset doesn't need to be split
    test_dataset = dataset.map(lambda x: to_input(cfg, x))
    test_dataset = dataset.batch(1)

    return train_dataset, test_dataset
    
