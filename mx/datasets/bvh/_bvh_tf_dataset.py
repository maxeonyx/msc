import abc
from ast import Tuple
from dataclasses import dataclass
from re import S
from tkinter import E
from typing import Collection, Literal, Optional, Union
from pathlib import Path

import tensorflow as tf
import einops as ein

from mx import utils
from .. import tasks
from .. import DSet
from mx.utils import tf_scope, Einshape
from . import _bvh

@dataclass
class _BvhCfg(abc.ABC):
    """
    Config for the BVH dataset pipeline.
    """
    recluster: bool = False
    decimate: Union[Literal[False], float] = False
    n_hands: Literal[1, 2] = 2

@dataclass
class BvhSpecificColumns(_BvhCfg):
    """
    Config for the BVH dataset pipeline.
    """
    n_dof_per_hand: int = 23
    columns: Literal["useful"] = "useful"

@dataclass
class BvhAllColumns(_BvhCfg):
    """
    Config for the BVH dataset pipeline.
    """
    n_joints_per_hand: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] = 17
    n_dof_per_joint: Literal[1, 2, 3] = 3
    columns: Literal["all"] = "all"


def _load_bvh_data(cfg: _BvhCfg) -> tuple[DSet, dict[str, Einshape]]:
    
    filenames, angles, n_frames = _bvh.np_dataset_parallel_lists(force=cfg.force, columns=cfg.columns)
    
    all_angles = tf.concat(angles, axis=0)
    del angles
    n_frames = tf.constant(n_frames)
    ragged_angles = tf.RaggedTensor.from_row_lengths(all_angles, n_frames)
    
    orig_angles = tf.data.Dataset.from_tensor_slices(ragged_angles)
    filenames = tf.data.Dataset.from_tensor_slices(filenames)

    dset = tf.data.Dataset.zip({ "filename": filenames, "orig_angles": orig_angles })
    n = dset.cardinality()

    dset = dset.shuffle(buffer_size=n, seed=1234)

    test_size = n // 10
    dset = DSet(
        test=dset.take(test_size),
        val=dset.skip(test_size).take(test_size),
        train=dset.skip(2 * test_size)
    )

    dset = dset.map(lambda x: { **x, "angles": x["orig_angles"] })
    def map_angles(dataset, fn):
        return dataset.map(lambda x: { **x, "angles": fn(x["angles"]) })

    if cfg.recluster:
        circular_means = utils.circular_mean(all_angles, axis=0)
        dset = map_angles(dset, lambda a: utils.recluster(a, circular_means))
    
    # subset
    if isinstance(cfg, BvhSpecificColumns):
        dset = map_angles(dset, lambda a: a[:, :cfg.n_hands :cfg.n_dof_per_hand])
        feature_dims = [
            ("h", cfg.n_hands),
            ("d", cfg.n_dof_per_hand),
        ]
    elif isinstance(cfg, BvhAllColumns):
        dset = map_angles(dset, lambda a: a[:, :cfg.n_hands, :cfg.n_joints_per_hand, :cfg.n_dof_per_joint])
        feature_dims = [
            ("h", cfg.n_hands),
            ("j", cfg.n_joints_per_hand),
            ("d", cfg.n_dof_per_joint),
        ]
    else:
        raise ValueError(f"Config type {type(cfg)} not implemented for BVH dataset")

    dset = dset.map(lambda x: { **x, "idxs": utils.multidim_indices(tf.shape(x["angles"])) })

    if cfg.decimate:
        index_dims = feature_dims + [("i", len(feature_dims))]
        decimate = make_decimate_fn(cfg.decimate, feature_dims, other_seqs=[index_dims])
        def do_decimate(x):
            angles, [idxs] = decimate(x["angles"], [x["idxs"]])
            return {
                **x,
                "angles": angles,
                "idxs": idxs,
            }
        dset = dset.map(do_decimate)
    
    dset = dset.snapshot(cfg.cached_dataset_path, compression=None)
    dset = dset.cache()

    angles_einshape = utils.Einshape(
        batch_dims={},
        seq_dims={ "f": "ragged" },
        feature_dims=dict(feature_dims),
    )
    orig_angles_einshape = utils.Einshape(
        batch_dims={},
        seq_dims={ "f": "ragged" },
        feature_dims=dict(feature_dims),
    )
    idxs_einshape = utils.Einshape(
        batch_dims={},
        seq_dims={ "f": "ragged" },
        feature_dims=dict(index_dims),
    )

    return dset, {
        "angles": angles_einshape,
        "orig_angles": orig_angles_einshape,
        "idxs": idxs_einshape,
    }


def vector_ntp(seq_dims: Collection[str], feat_dims: Collection[str], d_cfg: _BvhCfg, t_cfg: tasks.NextVectorPrediction) -> DSet:
    
    dset, shapes = _load_bvh_data(d_cfg)

    seq_dims = dict(seq_dims)
    feat_dims = dict(feat_dims)

    assert "f" in seq_dims, "Frame dimension must be a sequence dimension"
    assert len(feat_dims) >= 1, "At least one feature dimension must be present"

    if isinstance(d_cfg, BvhSpecificColumns):
        dims = ["f", "h", "d"]
    elif isinstance(d_cfg, BvhAllColumns):
        dims = ["f", "h", "j", "d"]
    else:
        raise ValueError(f"Config type {type(d_cfg)} not implemented for `vector_ntp`")

    for dim in dims:
        assert dim in seq_dims or dim in feat_dims, f"Dimension {dim} was not included in either sequence or feature dimensions"
    
    n_seq_dims = 1
    if "h" in seq_dims:
        n_seq_dims += 1
    if "j" in seq_dims:
        assert "h" in seq_dims, "Joint dimension 'j' must be after hand dimension 'h'"
        n_seq_dims += 1
    if "d" in seq_dims:
        assert "j" in seq_dims, "DOF dimension 'd' must be after joint dimension 'j'"
        n_seq_dims += 1
    
    assert n_seq_dims <= len(seq_dims), ("Sequence dimensions must be a prefix of the feature dimensions. Otherwise,"
        "use a different task function.")

    def do_chunk(x):

        angles, orig_angles, idxs = get_chunk(
            seqs=[
                (x["angles"], shapes["angles"]),
                (x["orig_angles"], shapes["orig_angles"]),
                (x["idxs"], shapes["idxs"]),
            ],
            chunk_size=[t_cfg.sequence_length],
            chunk_mode="simple",
        )

        return {
            **x,
            "angles": angles,
            "orig_angles": orig_angles,
            "idxs": idxs,
        }
    
    # chunk
    dset = dset.map(do_chunk)

    # repeat data to take many random chunks from each sequence
    train, test, val = dset.destructure()
    n_train = train.cardinality().numpy()
    dset = DSet(
        # repeat training data infinitely
        train=train.repeat().shuffle(n_train),

        # take 10 random chunks from each example
        test=test.repeat(10),
        val=val.repeat(10),
    )

    # set shapes to chunk size
    shapes = {
        "inputs": {
            "input": shapes["angles"].with_sequence_dims({ "f": t_cfg.sequence_length - 1 }),
            "idxs": shapes["idxs"].with_sequence_dims({ "f": t_cfg.sequence_length - 1 }),
            "target_idxs": shapes["idxs"].with_sequence_dims({ "f": t_cfg.sequence_length }),
        },
        "targets": {
            "target": shapes["angles"].with_sequence_dims({ "f": t_cfg.sequence_length }),
        },
        "extra": {
            "orig_angles": shapes["orig_angles"],
            "filename": (),
        },
    }

    dset = dset.map(lambda x: {
        "inputs": {
            "input": x["angles"][:, :-1],
            "idxs": x["idxs"][:, :-1],
            "target_idxs": x["idxs"],
        },
        "targets": {
            "target": x["angles"],
        },
        "extra": {
            "orig_angles": x["orig_angles"],
            "filename": x["filename"],
        },
    })

    return dset, shapes

def to_inputs_and_targets(x):

    input = x["angles"][:, :-1]
    input_idxs = x["idxs"][:, :-1]
    input_target_idxs = x["idxs"]

    target = x["angles"]

    return {
        "input": input,
        "input_idxs": input_idxs,
        "input_target_idxs": input_target_idxs,
    }, {
        "target": target,
    }, {
        
    }


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
def get_chunk(seqs: list[tuple[tf.Tensor, Einshape]], chunk_size: list[int], chunk_mode: Literal["simple", "random"], seed=None):
    """
    Cuts chunks of size chunk_size from the input sequence x.
    Returns a new sequence of the same rank as x, with the
    sequence dimensions being cut to chunk_size.

    Does not support batching. Use tf.map_fn to batch.

    Sequence dimensions can be ragged, in which case this
    function can be used to cut non-ragged chunks, and will
    return a non-ragged sequence of the same rank as x.
    """

    assert len(seqs) > 0, "Must provide at least one sequence"

    seqs = [ tf.ensure_shape(s, e.shape) for s, e in seqs ]

    assert [ e.s_shape == seqs[0][1].s_shape for s, e in seqs ], "All sequences must have the same sequence dimensions"
    seq_shape = seqs[0][1].s_shape
    seq_rank = len(seq_shape)

    assert len(chunk_size) == seq_rank, f"Chunk size {chunk_size} must have same rank as seq_dims {seq_shape}"
    assert all([s > 0 for s in chunk_size]), f"Chunk size {chunk_size} must be positive"
    assert all([s <= dim for s, dim in zip(chunk_size, seq_shape)]), f"Chunk size {chunk_size} must be smaller than seq_dims {seq_shape}"

    seqs = [
        (
            ein.rearrange(s, f'... {e.s_str} {e.f_str} -> ... {e.s_str} ({e.f_str})', **e.s_dict, **e.f_dict),
            e
        )
        for s, e in seqs
    ]
    
    if chunk_mode == "simple":
        max_start_i = tf.constant(seq_shape) - tf.constant(chunk_size)
        i = tf.random.uniform(
            [seq_rank], minval=tf.zeros([seq_rank]), maxval=max_start_i, seed=seed, dtype=tf.int32)
        idxs = tf.tile(i, [*seq_shape, 1]) + utils.multidim_indices(chunk_size, flatten=False)
    elif chunk_mode == "random":
        idxs = utils.multidim_indices(seq_shape, flatten=False)
        idxs = random_subset(idxs, chunk_size, seed=seed)
    else:
        raise ValueError(f"Unknown chunk mode {chunk_mode}.")

    # extract chunks from seqs
    seqs = [
        (
            tf.gather_nd(s, idxs),
            e,
        )
        for s, e in seqs
    ]

    # restore shape of feature dimensions
    seqs = [
        (
            ein.rearrange(s, f'... {e.s_str} ({e.f_str}) -> ... {e.s_str} {e.f_str}', **e.s_dict, **e.f_dict),
            e,
        )
        for s, e in seqs
    ]

    # ensure new shape (important for ragged tensors)
    seqs = [ tf.ensure_shape(s, e.shape) for s, e in seqs ]

    return seqs


def make_decimate_fn(threshold: float, feat_dims: list[tuple[str, int]], other_seqs: list[list[tuple[str, int]]] = []):
    """
    Cut down a sequence along a single `seq_dim` to the regions where the feature
    values vary, as measured by the L2 norm across `feat_dims`.

    Does not support batching, use tf.map_fn to apply to a batch.
    """

    feat_dims_shape = [d for _, d in feat_dims]
    feat_dims_str = " ".join([s for s, _ in feat_dims])
    feat_dims_dict = dict(feat_dims)

    other_seqs_shape = [[d for _, d in seq] for seq in other_seqs]
    other_seqs_str = [" ".join([s for s, _ in seq]) for seq in other_seqs]
    
    def decimate(data, other_data):
        data = ein.rearrange(data, f'f {feat_dims_str} -> f ({feat_dims_str})', **feat_dims_dict)
        other_data = [ein.rearrange(o_data, f'f {o_str} -> f ({o_str})', **o_dict) for o_data, o_str, o_dict in zip(other_data, other_seqs_str, other_seqs)]
        len_data = tf.shape(data)[0]
        decimated_data = data[:1, :]
        decimated_other_data = [o_data[:1, :] for o_data in other_data]
        for i in tf.range(1, len_data):
            if tf.linalg.norm(data[i] - decimated_data[-1]) > threshold:
                decimated_data = tf.concat([decimated_data, data[i:i+1]], axis=0)
                decimated_other_data = [tf.concat([o_data, o_data[i:i+1]], axis=0) for o_data in decimated_other_data]
        decimated_data = ein.rearrange(decimated_data, f'f ({feat_dims_str}) -> f {feat_dims_str}', **feat_dims_dict)
        decimated_other_data = [ein.rearrange(o_data, f'f ({o_str}) -> f {o_str}', **o_dict) for o_data, o_str, o_dict in zip(decimated_other_data, other_seqs_str, other_seqs)]
        return decimated_data, decimated_other_data
    
    if len(other_seqs) == 0:
        def decimate_fn(data):
            d, other_d = decimate(data, [])
            return d
        input_signature=[
            tf.TensorSpec(shape=[None, *feat_dims_shape], dtype=tf.float32),
        ]
    else:
        other_seqs_tspec = tuple([tf.TensorSpec(shape=[None, *s]) for s in other_seqs_shape])
        input_signature = [
            tf.TensorSpec(shape=[None, *feat_dims_shape], dtype=tf.float32),
            *other_seqs_tspec,
        ]

        def decimate_fn(data, other_data):
            return decimate(data, other_data)

    decimate_fn = tf.function(tf_scope(decimate_fn), input_signature=input_signature)
    
    return decimate_fn
