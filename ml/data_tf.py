"""
Takes a np BVH dataset and returns a model-agnostic tf.data.Dataset pipeline, which can be further transformed.
"""

import sys

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
    x["frame_idxs"] = x["frame_idxs"][:cfg.chunk_size, :, :]
    x["hand_idxs"] = x["hand_idxs"][idx:idx+cfg.chunk_size, :, :]
    x["dof_idxs"] = x["dof_idxs"][idx:idx+cfg.chunk_size, :, :]

    return x

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
def random_flat_chunk(cfg, x, aligned=False):
    shape = tf.shape(x["angles"])
    n_frames = shape[0]
    
    x["angles"] = tf.reshape(x["angles"], [-1])
    x["frame_idxs"] = tf.reshape(x["frame_idxs"], [-1])
    x["hand_idxs"] = tf.reshape(x["hand_idxs"], [-1])
    x["dof_idxs"] = tf.reshape(x["dof_idxs"], [-1])
    
    tok_per_frame = cfg.n_hands * cfg.n_dof
    chunk_size = cfg.chunk_size
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

# cut a fixed size chunk from the tensor at a random token index
def random_flat_chunk_batched(cfg, x, aligned=False):
    shape = tf.shape(x["angles"])
    batch_size = shape[0]
    n_frames = shape[1]

    x["angles"] = tf.reshape(x["angles"], [batch_size, -1])
    x["frame_idxs"] = tf.reshape(x["frame_idxs"], [batch_size, -1])
    x["hand_idxs"] = tf.reshape(x["hand_idxs"], [batch_size, -1])
    x["dof_idxs"] = tf.reshape(x["dof_idxs"], [batch_size, -1])
    
    tok_per_frame = cfg.n_hands * cfg.n_dof
    chunk_size = cfg.chunk_size
    chunk_toks = chunk_size * tok_per_frame

    if aligned:
        # get a random index
        idx = tf.random.uniform([batch_size], minval=0, maxval=n_frames-chunk_size, dtype=tf.int32)
        idx = idx * tok_per_frame
    else:
        idx = tf.random.uniform([batch_size], minval=0, maxval=n_frames*tok_per_frame-chunk_toks, dtype=tf.int32)

    batch_i = tf.stack([tf.tile(tf.range(batch_size)[:, None], [1, chunk_toks]), idx[:, None] + tf.range(chunk_toks)[None, :]], axis=-1)
    print(batch_i)
    x["angles"] = tf.gather_nd(x["angles"], batch_i)
    print(x["angles"])
    if cfg.relative_frame_idxs:
        x["frame_idxs"] = tf.vectorized_map(lambda i: frame_idxs_for(cfg, i % tok_per_frame, chunk_toks), idx)
    else:
        x["frame_idxs"] = tf.gather_nd(x["frame_idxs"], batch_i)


    x["hand_idxs"] = tf.gather_nd(x["hand_idxs"], batch_i)
    x["dof_idxs"] = tf.gather_nd(x["dof_idxs"], batch_i)

    return x

def to_train_input_and_target(cfg, x):
    inp = x.copy()

    inp["angles"] = inp["angles"][..., :-1]
    inp["frame_idxs"] = inp["frame_idxs"][..., :-1]
    inp["hand_idxs"] = inp["hand_idxs"][..., :-1]
    inp["dof_idxs"] = inp["dof_idxs"][..., :-1]

    if cfg.target_is_sequence:
        tar = x["angles"]
        tf.ensure_shape(tar, [cfg.chunk_size * cfg.n_hands * cfg.n_dof])
    else:
        tar = x["angles"][..., -1]

    return (inp, tar)

def to_test_input_and_target(cfg, x):
    inp = x.copy()

    seq_len = tf.shape(inp["angles"])[0]

    predict_tokens = cfg.predict_frames * cfg.n_hands * cfg.n_dof

    inp["angles"] = inp["angles"][:seq_len-predict_tokens]
    inp["frame_idxs"] = inp["frame_idxs"][:seq_len-predict_tokens]
    inp["hand_idxs"] = inp["hand_idxs"][:seq_len-predict_tokens]
    inp["dof_idxs"] = inp["dof_idxs"][:seq_len-predict_tokens]

    tar = x

    return (inp, tar)

def subset(cfg, x):
    """
    Slice just some of the data.
    """
    x["angles"] = x["angles"][:, :cfg.n_hands, :cfg.n_dof]
    x["frame_idxs"] = x["frame_idxs"][:, :cfg.n_hands, :cfg.n_dof]
    x["hand_idxs"] = x["hand_idxs"][:, :cfg.n_hands, :cfg.n_dof]
    x["dof_idxs"] = x["dof_idxs"][:, :cfg.n_hands, :cfg.n_dof]

    return x


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

    dataset = dataset.map(lambda x: subset(cfg, x))
    
    # train input isn't always frame aligned
    train_dataset = dataset.map(lambda x: random_flat_chunk(cfg, x))
    train_dataset = train_dataset.batch(cfg.batch_size)
    # split into input and target, with 1 token offset
    train_dataset = train_dataset.map(lambda x: to_train_input_and_target(cfg, x))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # test input is always frame-aligned
    # take fixed size chunks from the tensor at random frame indices
    test_dataset = dataset.map(lambda x: random_flat_chunk(cfg, x, aligned=True))
    test_dataset = test_dataset.map(lambda x: to_test_input_and_target(cfg, x))
    # test dataset doesn't need to be split
    test_dataset = test_dataset.batch(1)

    return train_dataset, test_dataset
    
