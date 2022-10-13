from dataclasses import dataclass
from os import PathLike
from typing import Any, Callable, Literal, Union

import holoviews as hv

from mx.prelude import *
from mx.tasks import NextUnitVectorPrediction
from mx import utils as u
from mx.visualizer import HoloMapVisualization, StatefulVisualization
from mx.utils import Einshape, DSets
from mx.pipeline import MxDataset, Task

from . import _bvh

@export
@dataclass
class BvhDataset(MxDataset):
    """
    Config for the BVH dataset pipeline.
    """

    def __init__(
        self,
        recluster: bool = False,
        decimate: Union[Literal[False], float] = False,
        n_hands: Literal[1, 2] = 2,
        n_joints_per_hand: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] = 17,
        n_dof_per_joint: Literal[1, 2, 3] = 3,
        name="BVH dataset",
        identifier="bvh",
        split=(0.8, 0.1, 0.1),
        split_seed=1234,
    ):
        super().__init__(
            desc=name,
            name=identifier,
            split=split,
            split_seed=split_seed,
        )
        self.recluster = recluster
        """Whether to recluster the angles to have a circular mean of 0."""
        self.decimate = decimate
        """
        Whether to decimate the dataset. If a float, this is min L2-dist
        between two frames.
        """
        self.n_hands = n_hands
        """Number of hands to use."""
        self.n_joints_per_hand = n_joints_per_hand
        """Number of joints per hand to use."""
        self.n_dof_per_joint = n_dof_per_joint
        """Number of degrees of freedom per joint to use."""

        self.n_features = self.n_hands * self.n_joints_per_hand * self.n_dof_per_joint
        """Number of features when (hands joints dof) dims are flattened."""

    def configure(self, task: Task):

        # vector of shape (h j d)
        n_input_dims = self.n_hands * self.n_joints_per_hand * self.n_dof_per_joint

        if isinstance(task, NextUnitVectorPrediction):
            task.recieve_dataset_config(task.ds_config_type(
                n_input_dims=n_input_dims,
            ))
            def adapt_in(x):
                return {
                    # flatten hand/joint/dof dims into a single feature dim
                    "data": ein.rearrange(x["angles"], f"... f h j d -> ... f (h j d)"),
                    # remove indices for the flattened feature dim, keep only
                    # frame indices
                    "seq_idxs": x["idxs"][..., :, 0, 0, 0, 0],
                    "extra": x["extra"],
                }
            self.adapt_in = adapt_in
            def adapt_out(x):
                # adapt output from a task's predict_fn into dataset-specific format
                return {
                    "angles": (
                        "Predicted Angles",
                        ein.rearrange(
                            x["angles"],
                            f"... f (h j d) -> ... f h j d",
                            h=self.n_hands,
                            j=self.n_joints_per_hand,
                            d=self.n_dof_per_joint
                        ),
                    )
                }
            self.adapt_out = adapt_out
        else:
            raise NotImplementedError(f"{type(task)} not supported by {type(self)}")

        assert self.adapt_in is not None, "Forgot to set self.adapt_in"
        assert self.adapt_out is not None, "Forgot to set self.adapt_out"


    def load(self, force_cache_reload: bool) -> DSets:

        assert self.adapt_in is not None, "Must call dataset.configure(task) before dataset.load()"

        filenames, angles, n_frames = _bvh.np_dataset_parallel_lists(force=force_cache_reload, columns="all")

        all_angles = tf.concat(angles, axis=0)
        del angles

        all_angles = tf.reshape(all_angles, [-1, 2, 17, 3])
        all_angles = all_angles[:, :self.n_hands, :self.n_joints_per_hand, :self.n_dof_per_joint]
        angles_einshape = Einshape(
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
            circular_means = u.circular_mean(all_angles, axis=0)
            dset = map_angles(dset, lambda a: u.recluster(a, circular_means))

        dset = dset.map(lambda x: { **x, "idxs": u.multidim_indices_of(x["angles"], flatten=False) })
        index_einshape = angles_einshape.append_feature_dim("i", angles_einshape.rank)

        if self.decimate:
            decimate = make_decimate_fn(self.decimate, angles_einshape, other_params=[(index_einshape, tf.int32)])
            def do_decimate(x):
                angles, [idxs] = decimate(x["angles"], x["idxs"])
                return {
                    "angles": angles,
                    "idxs": idxs,
                    "extra": x["extra"],
                }
            dset = dset.map(do_decimate)

        dsets = self._snapshot_and_split(dset, buffer_size=None)

        return dsets

    def get_visualizations(self, viz_batch_size, task_specific_predict_fn) -> dict[str, StatefulVisualization]:

        assert self.adapt_in is not None, "Must call dataset.configure(task) before dataset.get_visualizations()"

        viz_data = (
            self.load(force_cache_reload=False)
            .test
            .take(viz_batch_size)
        )
        viz_data = [ d for d in viz_data ]
        #  # add batch dim, tbc how i'm gonna do batching because i need to support ragged
        #  # sequence dims and einops doesn't like RaggedTensor
        # viz_data = tf.expand_dims(viz_data, axis=0)

        return u.list_to_dict([
            BVHImageViz(
                data=viz_data,
                task_predict_fn=task_specific_predict_fn,
                adapt_in=self.adapt_in,
                adapt_out=self.adapt_out,
            ),
        ])

@export
class BVHImageViz(HoloMapVisualization):

    def __init__(self,
        data: list[dict[str, tf.Tensor | dict[str, tf.Tensor]]],
        task_predict_fn: Callable[[dict[str, tf.Tensor]], dict[str, tf.Tensor]],
        adapt_in: Callable,
        adapt_out: Callable,
        name = "bvh_imgs",
        desc = "BVH Image Viz",
    ):
        super().__init__(
            name=name,
            desc=desc,
        )
        self._data = data
        self._task_predict_fn = task_predict_fn
        self._adapt_in = adapt_in
        self._adapt_out = adapt_out

    def make_hmaps(self, i_step) -> list[hv.HoloMap]:
        data = self._data
        # data comes from dsets.test in "dataset" format

        assert isinstance(data, list),                                       f"data must be a list of batches. Got: type(data)={type(data).__name__}"
        batch_size = len(data)
        assert isinstance(data[0], dict),                                    f"data must be a list of batches, each one a dict. Got: type(data[0])={type(data[0]).__name__}"
        assert "angles" in data[0],                                          f"data[0] must contain 'angles'. Got keys: [{data[0].keys()}]"
        assert tf.is_tensor(data[0]["angles"]),                              f"data[0]['angles'] must be a tensor. Got: type(data[0]['angles'])={type(data[0]['angles']).__name__}"
        assert data[0]["angles"].dtype == tf.float32,                         "data[0]['angles'] must be a float32 tensor"
        assert data[0]["angles"].shape.rank == 4,                            f"data[0]['angles'] must be a float32 tensor of shape [b f h j d]. Got {data[0]['angles'].shape}"
        assert "idxs" in data[0],                                             "data[0] must contain 'idxs'"
        assert tf.is_tensor(data[0]["idxs"]),                                 "data[0]['idxs'] must be a tensor"
        assert data[0]["idxs"].dtype == tf.int32,                             "data[0]['idxs'] must be a int32 tensor"
        assert data[0]["idxs"].shape.rank == 5,                              f"data[0]['idxs'] must be a int32 tensor of shape [b f h j d i]. Got {data[0]['idxs'].shape}"
        assert isinstance(data[0]["extra"], dict),                            "data[0]['extra'] must be a dict"
        assert "orig_angles" in data[0]["extra"],                             "data[0]['extra'] must contain 'orig_angles'"
        assert tf.is_tensor(data[0]["extra"]["orig_angles"]),                f"data[0]['extra']['orig_angles'] must be a tensor. Got: type(data[0]['extra']['orig_angles'])={type(data[0]['extra']['orig_angles']).__name__}"
        assert data[0]["extra"]["orig_angles"].dtype == tf.float32,           "data[0]['extra']['orig_angles'] must be a float32 tensor"
        assert data[0]["extra"]["orig_angles"].shape.rank == 4,              f"data[0]['extra']['orig_angles'] must be a float32 tensor of shape [b f h j d]. Got {data[0]['extra']['orig_angles'].shape}"
        assert "filename" in data[0]["extra"],                                "data[0]['extra'] must contain 'filename'"
        assert tf.is_tensor(data[0]["extra"]["filename"]),                    "data[0]['extra']['filename'] must be a tensor"
        assert data[0]["extra"]["filename"].dtype == tf.string,               "data[0]['extra']['filename'] must be a string tensor"

        task_specific_input_data = [ self._adapt_in(d) for d in data ]
        predict_outputs = [
            self._adapt_out(
                self._task_predict_fn(d)
            )
            for d in task_specific_input_data
        ]

        # use tf.nest.map_structure to stack all component tensors
        predict_outputs = tf.nest.map_structure(
            lambda *xs: tf.concat(xs, axis=0),
            *predict_outputs,
        )

        assert isinstance(predict_outputs, dict), "predict_outputs must be a dict"

        for ident, o in predict_outputs.items():
            assert isinstance(o, tuple),           f"predict_outputs['{ident}'] must be a tuple"
            assert len(o) == 2,                    f"predict_outputs['{ident}'] must be a tuple of length 2"
            name, v = o

            # assert isinstance(name, str),          f"predict_outputs['{ident}'][0] must be a string"
            assert tf.is_tensor(name),             f"predict_outputs['{ident}'][0] must be a tensor"
            assert name.dtype == tf.string,        f"predict_outputs['{ident}'][0] must be a string tensor"

            assert tf.is_tensor(v),                f"predict_outputs['{ident}'][1] must be a tensor. Got type={type(v).__name__}"
            assert v.dtype == tf.float32,          f"predict_outputs['{ident}'][1] must be a float32 tensor. Got {v.dtype}"
            assert v.shape.rank == 5,              f"predict_outputs['{ident}'][1] must be a float32 tensor of shape [b f h j d]. Got {v.shape}"


        key_dims = [
            hv.Dimension(("step", "Step")),
            hv.Dimension(("batch", "Batch")),
        ]
        def img(data, title):
            data = ein.rearrange(data, "f h j d -> (h j d) f")
            return hv.Raster(data.numpy()).opts(
                title=title,
                cmap='twilight',
                aspect='equal',
            )

        return [
            hv.HoloMap(
                {
                    (i_step, i_batch): img(data[i_batch], u.tf_str(titles[i_batch]))
                    for i_batch in range(batch_size)
                },
                kdims=key_dims,
                label=ident
            )
            for ident, (titles, data) in predict_outputs.items()
        ]

def random_subset(options, n, seed=None):
    options = tf.random.shuffle(options, seed=seed)
    return options[:n]

@u.tf_scope
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
    @u.tf_scope
    def get_chunk(seqs):
        seqs = [ tf.ensure_shape(s, e.shape) for s, e in zip(seqs, seq_dims) ]
        seqs = [
            ein.rearrange(s, f'... {e.f_str} -> ... ({e.f_str})', **e.f_dict)
            for s, e in zip(seqs, seq_dims)
        ]

        seq_shape = tf.shape(seqs[0])[seq_einshape.b_rank:seq_einshape.b_rank+seq_einshape.s_rank]

        if chunk_mode == "simple":

            max_indices = seq_shape - tf.constant(chunk_size, tf.int32)
            idxs = tf.map_fn(
                lambda max_i: tf.random.uniform([], 0, max_i, dtype=tf.int32, seed=seed),
                max_indices,
            )
            idxs = idxs[None, :] + u.multidim_indices(chunk_size, flatten=True, elide_rank_1=False)
        elif chunk_mode == "random":
            idxs = u.multidim_indices(seq_shape, flatten=False)
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
            ein.rearrange(s, f'... ({e.s_str}) ({e.f_str}) -> ... {e.s_str} {e.f_str}', **e.s_dict, **e.f_dict)
            for s, e in zip(seqs, new_seq_dims)
        ]

        # ensure new shape (important for ragged tensors)
        seqs = [ tf.ensure_shape(s, e.shape) for s, e in zip(seqs, new_seq_dims) ]

        return seqs

    return get_chunk

def make_decimate_fn(threshold: float, dims: Einshape, other_params: list[tuple[Einshape, ...]] = []):
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
    @u.tf_scope
    def decimate(data, *other_data):
        data = ein.rearrange(data, f'f {dims.f_str} -> f ({dims.f_str})', **dims.f_dict)
        other_data = [ein.rearrange(o_data, f'f {o.f_str} -> f ({o.f_str})', **o.f_dict) for o_data, o in zip(other_data, other_dims)]
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
                tf.TensorShape([None, *shape(data)[1:]]),
                [tf.TensorShape([None, *shape(o_data)[1:]]) for o_data in other_data],
            ],
        )

        decimated_data = ein.rearrange(decimated_data, f'f ({dims.f_str}) -> f {dims.f_str}', **dims.f_dict)
        decimated_other_data = [ein.rearrange(o_data, f'f ({o.f_str}) -> f {o.f_str}', **o.f_dict) for o_data, o in zip(decimated_other_data, other_dims)]
        return decimated_data, decimated_other_data

    if len(other_params) == 0:
        def decimate(data):
            d, _other_d = decimate(data)
            return d

    return decimate


if __name__ == '__main__':
    import doctest
    doctest.testmod()
