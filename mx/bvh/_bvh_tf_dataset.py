import abc
from ast import Tuple
from dataclasses import dataclass
from re import S
from typing import Literal, Optional, Union
from pathlib import Path

import tensorflow as tf
import einops as ein

from mx import utils, tasks, datasets as ds
from mx.utils import tf_scope
from . import _bvh

@dataclass
class BvhCfg(abc.ABC):
    """
    Config for the BVH dataset pipeline.
    """
    force: bool = False
    recluster: bool = False
    decimate: Union[Literal[False], float] = False
    n_hands: Literal[1, 2] = 2
    cached_dataset_path: str = Path.cwd() / "_cache/bvh_dataset"

@dataclass
class BvhSpecificColumns(BvhCfg):
    """
    Config for the BVH dataset pipeline.
    """
    n_dof_per_hand: int = 23
    columns: Literal["useful"] = "useful"

@dataclass
class BvhAllColumns(BvhCfg):
    """
    Config for the BVH dataset pipeline.
    """
    n_joints_per_hand: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] = 17
    n_dof_per_joint: Literal[1, 2, 3] = 3
    columns: Literal["all"] = "all"

class BVHDataset:
    """
    BVH dataset pipeline.
    """
    def __init__(self, cfg: BvhCfg):
        self.cfg = cfg
    
        filenames, angles, n_frames = _bvh.np_dataset_parallel_lists(force=cfg.force, columns=cfg.columns)
        
        all_angles = tf.concat(angles, axis=0)
        n_frames = tf.constant(n_frames)
        ragged_angles = tf.RaggedTensor.from_row_lengths(all_angles, n_frames)
        
        orig_angles = tf.data.Dataset.from_tensor_slices(ragged_angles)
        filenames = tf.data.Dataset.from_tensor_slices(filenames)

        raw_data = tf.data.Dataset.zip({ "filename": filenames, "orig_angles": orig_angles })
        n = raw_data.cardinality()

        raw_data = raw_data.cache()
        raw_data = raw_data.shuffle(buffer_size=n, seed=1234)
        raw_data = raw_data.snapshot(cfg.cached_dataset_path, compression=None)

        test_size = n // 10
        n_train = n - (2 * test_size)
        raw_dset = ds.DSet(
            test=raw_data.take(test_size),
            val=raw_data.skip(test_size).take(test_size),
            train=(
                raw_data
                    .skip(2 * test_size)
                    .repeat()
                    .shuffle(buffer_size=n_train),
            )
        )

        dset = raw_dset.map(lambda x: x["orig_angles"])

        if cfg.recluster:
            circular_means = utils.circular_mean(all_angles, axis=0)
            dset = dset.map(lambda x: utils.recluster(x, circular_means))
        
        # subset
        if isinstance(cfg, BvhSpecificColumns):
            dset = dset.map(lambda a: a[:, :cfg.n_hands :cfg.n_dof_per_hand])
        elif isinstance(cfg, BvhAllColumns):
            dset = dset.map(lambda a: a[:, :cfg.n_hands, :cfg.n_joints_per_hand, :cfg.n_dof_per_joint])
        else:
            raise ValueError(f"Config type {type(cfg)} not implemented for BVH dataset")

        if cfg.decimate:
            decimate = make_decimate_fn(cfg.decimate, {
                "h": cfg.n_hands,
                "j": cfg.n_joints_per_hand,
                "d": cfg.n_dof_per_joint,
            })
            dset = dset.map(decimate)

        self.raw_dset: tf.data.Dataset = raw_dset
        self.dsets: ds.DSet = dset

    def task(self, task: tasks.TaskCfg):
        
        if isinstance(task, tasks.NextTokenPrediction):
            pass
        
        raise ValueError(f"Task type {type(task)} not implemented for BVH dataset")


def random_subset(options, n, seed=None):
    options = tf.random.shuffle(options, seed=seed)
    return options[:n]

@tf_scope
def weighted_random_n(n_max, D=2., B=10., seed=None):
    """
    Picks random indices from [0, n_max) with skewed distribution
    towards the lower numbers.
    p(index i) = 1/D * p(index Di) = 1/D**2 * p(index D**2i)
    The probability of each index decreases by D every power of b
    """

    B, D, n_max = tf.cast(B, tf.float32), tf.cast(
        D, tf.float32), tf.cast(n_max, tf.float32)

    def logb1p(b, x): return tf.math.log1p(x)/tf.math.log(b)
    # probabilities
    # probabilities = l1_norm(powf(D, -logb1p(B, tf.range(n_max))))
    # log probabilities base e
    log_probabilities = -logb1p(B, tf.range(n_max))*tf.math.log(D)

    return tf.random.categorical(log_probabilities[None, :], 1, seed=seed)[0, 0]

def flat_vector_batched_random_chunk(cfg, x, random_ahead=False, seed=None):

    if random_ahead:
        n_ahead = weighted_random_n(20, 2., 3., seed=seed)
    else:
        n_ahead = tf.constant(1, tf.int32)

    chunk_size = cfg.n_hand_vecs + n_ahead

    vecs, idxs = tf.map_fn(
        lambda a: get_chunk(cfg, chunk_size, a, chunk_mode="simple", seed=seed),
        x,
        fn_output_signature=(
            tf.TensorSpec(shape=[None, cfg.n_joints_per_hand,
                        cfg.n_dof_per_joint], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
        ),
        name="hand_map",
        parallel_iterations=10,
    )

    n_frames = tf.shape(vecs)[1]
    input = vecs[:, :n_frames-n_ahead, :, :]
    input_idxs = idxs[:, :n_frames-n_ahead, :]
    target = vecs[:, n_ahead:, :, :]
    target_idxs = idxs[:, n_ahead:, :]

    input = ein.rearrange(input, 'b fh j d -> b fh j d')
    target = ein.rearrange(target, 'b fh j d -> b (fh j d)')

    return (
        x | {
            "input_angles": input,
            "input_idxs": input_idxs,
            "target_idxs": target_idxs,
            # "n_ahead": n_ahead,
        },
        {
            "target_output": target,
        },
    )



@tf_scope
def get_chunk(x, chunk_size, chunk_mode: Literal["simple", "random"], dims: ds.Einshape, seed=None):
    """
    Cuts chunks of size chunk_size from the input sequence x.
    Returns a new sequence of the same rank as x, with the
    sequence dimensions being cut to chunk_size.

    Sequence dimensions can be ragged, in which case this
    function can be used to cut non-ragged chunks, and will
    return a non-ragged sequence of the same rank as x.
    """

    x = tf.ensure_shape(x, dims.shape)
    assert len(chunk_size) == dims.s_rank, f"Chunk size {chunk_size} must have same rank as seq_dims {dims.s_shape}"
    assert all([s > 0 for s in chunk_size]), f"Chunk size {chunk_size} must be positive"
    assert all([s <= dim for s, dim in zip(chunk_size, dims.s_shape)]), f"Chunk size {chunk_size} must be smaller than seq_dims {dims.s_shape}"

    x = ein.rearrange(x, f'... {dims.s_str} {dims.f_str} -> ... {dims.s_str} ({dims.f_str})', **dims.s, **dims.f)
    
    # sequence shape (support ragged sequences)
    seq_shape = tf.shape(x)[dims.b_rank:dims.b_rank+dims.s_rank]

    seq_idxs = utils.multidim_indices(dims.s_shape, flatten=False)
    
    if chunk_mode == "simple":
        i = tf.random.uniform(
            [dims.s_rank], minval=tf.zeros([dims.s_rank]), maxval=seq_shape-chunk_size, seed=seed, dtype=tf.int32)
        idxs = i + utils.multidim_indices(chunk_size, flatten=False)
    elif chunk_mode == "random":
        idxs = random_subset(seq_idxs, chunk_size, seed=seed)
    else:
        raise ValueError(f"Unknown chunk mode {chunk_mode}.")
    
    chunked_x = tf.gather_nd(x, idxs, batch_dims=dims.b_rank)
    chunked_x = ein.rearrange(chunked_x, f'... {dims.s_str} ({dims.f_str}) -> ... {dims.s_str} {dims.f_str}', **dims.s, **dims.f)

    dims = dims.cut(chunk_size)

    chunked_x = tf.ensure_shape(chunked_x, dims.shape)

    return chunked_x, idxs


def make_decimate_fn(threshold: float, dims: dict[str, int]):
    
    dims_shape = [d for d in dims.values()]
    dims_str = " ".join(dims.keys())

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, *dims_shape], dtype=tf.float32),
    ])
    @tf_scope
    def decimate(angles):
        angles = ein.rearrange(angles, 'f {dims_str} -> f ({dims_str})', **dims)
        len_angles = tf.shape(angles)[0]
        new_angles = angles[:1, :]
        for i in tf.range(1, len_angles):
            if tf.linalg.norm(angles[i] - new_angles[-1]) > threshold:
                new_angles = tf.concat([new_angles, angles[i:i+1]], axis=0)
        new_angles = ein.rearrange(new_angles, f'f ({dims_str}) -> f {dims_str}', **dims)
        return new_angles

    return decimate
