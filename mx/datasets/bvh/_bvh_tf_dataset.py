import abc
from dataclasses import dataclass
from typing import Any, Collection, Literal, Optional, Type, Union

from mx.utils.tf import *

from .. import tasks as orig_tasks
from mx import tasks
from mx.tasks import Task, NextUnitVectorPrediction
from .. import DSet, DatasetShape, Dataset, DsToTaskAdaptor, DSets
from mx import utils
from mx.utils import tf_scope, Einshape, shape_list
from . import _bvh

@dataclass
class BvhDataset(Dataset):
    """
    Config for the BVH dataset pipeline.
    """
    recluster: bool = False
    decimate: Union[Literal[False], float] = False
    n_hands: Literal[1, 2] = 2
    n_joints_per_hand: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] = 17
    n_dof_per_joint: Literal[1, 2, 3] = 3
    columns: Literal["all"] = "all"
    name = "bvh_all_columns"

    @property
    def implementations(self) -> dict[Type[Task], DsToTaskAdaptor]:
        return {
            NextUnitVectorPrediction: self._for_next_vector_prediction,
        }

    def _for_next_vector_prediction(self, dsets: DSets) -> DSets:

        dsets = dsets.map(lambda x: {
            # flatten hand/joint/dof dims into a single feature dim
            "data": ein.rearrange(x["angles"], f"f h j d -> f (h j d)"),
            # remove indices for the flattened feature dim, keep only
            # frame indices
            "seq_idxs": x["idxs"][:, 0, 0, 0, 0],
            "extra": x["extra"],
        })

        return dsets

    def load(self, force_cache_reload: bool) -> tf.data.Dataset:
        
        filenames, angles, n_frames = _bvh.np_dataset_parallel_lists(force=force_cache_reload, columns=self.columns)

        all_angles = tf.concat(angles, axis=0)
        del angles
        
        all_angles = tf.reshape(all_angles, [-1, 2, 17, 3])
        all_angles = all_angles[:, :self.n_hands, :self.n_joints_per_hand, :self.n_dof_per_joint]
        angles_einshape = utils.Einshape(
            batch_dims={},
            sequence_dims={ "f": "ragged" },
            feature_dims={
                "h": self.n_hands,
                "j": self.n_joints_per_hand,
                "d": self.n_dof_per_joint,
            },
        )
        
        n_frames = tf.constant(n_frames)
        ragged_angles = tf.RaggedTensor.from_row_lengths(all_angles, n_frames)
        
        orig_angles = tf.data.Dataset.from_tensor_slices(ragged_angles)

        filenames = tf.data.Dataset.from_tensor_slices(filenames)
        
        dset = tf.data.Dataset.zip((filenames, orig_angles))

        dset = dset.map(lambda filename, orig_angles: {
            "angles": orig_angles,
            "extra": {
                "filename": filename,
                "orig_angles": orig_angles,
            },
        })

        def map_angles(dataset, fn):
            return dataset.map(lambda x: { **x, "angles": fn(x["angles"]) })

        if self.recluster:
            circular_means = utils.circular_mean(all_angles, axis=0)
            dset = map_angles(dset, lambda a: utils.recluster(a, circular_means))

        dset = dset.map(lambda x: { **x, "idxs": utils.multidim_indices_of(x["angles"], flatten=False) })
        index_einshape = angles_einshape.append_feature_dim("i", angles_einshape.rank)

        if self.decimate:
            decimate = make_decimate_fn(self.decimate, angles_einshape, other_params=[(index_einshape, tf.int32)])
            def do_decimate(x):
                angles, [idxs] = decimate(x["angles"], x["idxs"])
                return {
                    **x,
                    "angles": angles,
                    "idxs": idxs,
                }
            dset = dset.map(do_decimate)

        return dset

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


def make_get_chunk(seq_dims: list[Einshape], chunk_size: list[int], chunk_mode: Literal["simple", "random"], seed=None):
    """
    Cuts chunks of size chunk_size from the input sequence x.
    Returns a new sequence of the same rank as x, with the
    sequence dimensions being cut to chunk_size.

    Does not support batching. Use tf.map_fn to batch.

    Sequence dimensions can be ragged, in which case this
    function can be used to cut non-ragged chunks, and will
    return a non-ragged sequence of the same rank as x.

    >>> x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> get_chunk = make_get_chunk([Einshape(sequence_dims={"a":3, "b":3})], [2, 2], chunk_mode="simple")
    >>> c = get_chunk([x])
    >>> any(tf.reduce_all(c) for c in [
    ...     tf.equal(c, tf.constant([[1, 2], [4, 5]])),
    ...     tf.equal(c, tf.constant([[2, 3], [5, 6]])),
    ...     tf.equal(c, tf.constant([[4, 5], [7, 8]])),
    ...     tf.equal(c, tf.constant([[5, 6], [8, 9]])),
    ... ])
    True
    """

    assert len(seq_dims) > 0, "Must provide at least one sequence"
    
    assert all([ e.b_shape == seq_dims[0].b_shape for e in seq_dims ]), "All sequences must have the same batch dimensions"
    assert all([ e.s_shape == seq_dims[0].s_shape for e in seq_dims ]), "All sequences must have the same sequence dimensions"
    seq_einshape = seq_dims[0]
    assert len(chunk_size) == seq_einshape.s_rank, f"Chunk size {chunk_size} must have same rank as seq_dims {seq_einshape.s_shape}"
    assert all([s > 0 for s in chunk_size]), f"Chunk size {chunk_size} must be positive"
    assert all([seq_dim is None or chunk_dim <= seq_dim for chunk_dim, seq_dim in zip(chunk_size, seq_einshape.s_shape)]), f"All dims of chunk size ({chunk_size}) must be <= seq_dims ({seq_einshape.s_shape})"

    @tf.function
    @tf_scope
    def get_chunk(seqs):
        seqs = [ tf.ensure_shape(s, e.shape) for s, e in zip(seqs, seq_dims) ]
        seqs = [
            ein.rearrange(s, f'... {e.f_str} -> ... ({e.f_str})', **e.f)
            for s, e in zip(seqs, seq_dims)
        ]

        seq_shape = tf.shape(seqs[0])[seq_einshape.b_rank:seq_einshape.b_rank+seq_einshape.s_rank]

        if chunk_mode == "simple":

            max_indices = seq_shape - tf.constant(chunk_size, tf.int32)
            idxs = tf.map_fn(
                lambda max_i: tf.random.uniform([], 0, max_i, dtype=tf.int32, seed=seed),
                max_indices,
            )
            idxs = idxs[None, :] + utils.multidim_indices(chunk_size, flatten=True, elide_rank_1=False)
        elif chunk_mode == "random":
            idxs = utils.multidim_indices(seq_shape, flatten=False)
            idxs = random_subset(idxs, chunk_size, seed=seed)
        else:
            raise ValueError(f"Unknown chunk mode {chunk_mode}.")

        # extract chunks from seqs
        seqs = [
            tf.gather_nd(s, idxs)
            for s in seqs
        ]

        new_seq_dims = [
            e.cut(chunk_size)
            for e in seq_dims
        ]

        # restore shape of sequence and feature dimensions
        seqs = [
            ein.rearrange(s, f'... ({e.s_str}) ({e.f_str}) -> ... {e.s_str} {e.f_str}', **e.s, **e.f)
            for s, e in zip(seqs, new_seq_dims)
        ]

        # ensure new shape (important for ragged tensors)
        seqs = [ tf.ensure_shape(s, e.shape) for s, e in zip(seqs, new_seq_dims) ]

        return seqs
    
    return get_chunk

def make_decimate_fn(threshold: float, dims: Einshape, other_params: list[tuple[Einshape, Any]] = []):
    """
    Cut down a sequence along a single `seq_dim` to the regions where the feature
    values vary, as measured by the L2 norm across `feat_dims`.

    Does not support batching, use tf.map_fn to apply to a batch.
    """

    other_dims = [d for d, t in other_params]

    if len(other_dims) == 0:
        input_signature=[
            tf.TensorSpec(shape=dims.s_f_shape, dtype=tf.float32),
        ]
    else:
        other_seqs_tspec = tuple([tf.TensorSpec(shape=d.s_f_shape, dtype=t) for d, t in other_params])
        input_signature = [
            tf.TensorSpec(shape=dims.s_f_shape, dtype=tf.float32),
            *other_seqs_tspec,
        ]
    
    @tf.function(input_signature=input_signature)
    @tf_scope
    def decimate(data, *other_data):
        data = ein.rearrange(data, f'f {dims.f_str} -> f ({dims.f_str})', **dims.f)
        other_data = [ein.rearrange(o_data, f'f {o.f_str} -> f ({o.f_str})', **o.f) for o_data, o in zip(other_data, other_dims)]
        len_data = tf.shape(data)[0]
        decimated_data = data[:1, :]
        decimated_other_data = [o_data[:1, :] for o_data in other_data]

        def decimate_step(i, decimated_data, decimated_other_data):
            decimated_data, decimated_other_data = tf.cond(
                pred=tf.greater(tf.linalg.norm(data[i] - decimated_data[-1]), threshold),
                true_fn=lambda: (
                    tf.concat([decimated_data, data[i:i+1]], axis=0),
                    [tf.concat([dec_o_data, o_data[i:i+1]], axis=0) for dec_o_data, o_data in zip(decimated_other_data, other_data)]
                ),
                false_fn=lambda: (decimated_data, decimated_other_data),
            )
            return i+1, decimated_data, decimated_other_data

        _i, decimated_data, decimated_other_data = tf.while_loop(
            lambda i, decimated_data, decimated_other_data: tf.less(i, len_data),
            decimate_step,
            [tf.constant(1), decimated_data, decimated_other_data],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, *utils.shape_list(data)[1:]]),
                [tf.TensorShape([None, *utils.shape_list(o_data)[1:]]) for o_data in other_data],
            ],
        )
            
        decimated_data = ein.rearrange(decimated_data, f'f ({dims.f_str}) -> f {dims.f_str}', **dims.f)
        decimated_other_data = [ein.rearrange(o_data, f'f ({o.f_str}) -> f {o.f_str}', **o.f) for o_data, o in zip(decimated_other_data, other_dims)]
        return decimated_data, decimated_other_data
    
    if len(other_params) == 0:
        def decimate(data):
            d, _other_d = decimate(data)
            return d
    
    return decimate

if __name__ == '__main__':
    import doctest
    doctest.testmod()
