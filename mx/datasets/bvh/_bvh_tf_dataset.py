from dataclasses import dataclass
from os import PathLike
from typing import Any, Callable, Literal, Union

import holoviews as hv
from mx.datasets.utils import make_decimate

from mx.prelude import *
from mx.tasks import VectorSequenceAngleMSE
from mx import utils as u
from mx.visualizer import HoloMapVisualization, StatefulVisualization, Visualization
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
        desc="BVH dataset",
        name="bvh",
        split=(0.8, 0.1, 0.1),
        split_seed=1234,
    ):
        super().__init__(
            desc=desc,
            name=name,
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

        self.split = split
        "Ratios of train/test/val split"

        self.split_seed = split_seed
        "Change this to split different data into train/test/val sets"


        self.n_features = self.n_hands * self.n_joints_per_hand * self.n_dof_per_joint
        """Number of features when (hands joints dof) dims are flattened."""

    def configure(self, task: Task):

        # vector of shape (h j d)
        n_input_dims = self.n_hands * self.n_joints_per_hand * self.n_dof_per_joint

        if isinstance(task, VectorSequenceAngleMSE):
            task.recieve_dataset_config(task.ds_config_cls(
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
            raise NotImplementedError(f"Dataset {type_name(self)} does not support Task {type_name(task)}. If using autoreload in IPython, try restarting the interpreter.")

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
            decimate = make_decimate(self.decimate, angles_einshape, other_params=[(index_einshape, tf.int32)])
            def do_decimate(x):
                angles, [idxs] = decimate(x["angles"], x["idxs"])
                return {
                    "angles": angles,
                    "idxs": idxs,
                    "extra": x["extra"],
                }
            dset = dset.map(do_decimate)

        dsets = self._snapshot_and_split(dset)

        return dsets

    def get_visualizations(self, viz_batch_size, task_specific_predict_fn) -> dict[str, Visualization]:

        assert self.adapt_in is not None, "Must call dataset.configure(task) before dataset.get_visualizations()"

        viz_data = (
            self.load(force_cache_reload=False)
            .test
            .batch(1)
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
        output_dir=None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            output_dir=output_dir,
        )

        # validate data in "dataset" format
        u.validate(data, "data", [
            {
                "angles": tf.TensorSpec(shape=[None, None, None, None], dtype=u.dtype()),
                "idxs": tf.TensorSpec(shape=[None, None, None, None, None], dtype=tf.int32),
                "extra": {
                    "orig_angles": tf.TensorSpec(shape=[None, None, None, None], dtype=u.dtype()),
                    "filename": tf.TensorSpec(shape=[], dtype=tf.string),
                },
            }
        ])
        self._data = data
        self._task_predict_fn = task_predict_fn
        self._adapt_in = adapt_in
        self._adapt_out = adapt_out

    def _make_hmaps(self, i_step) -> list[hv.HoloMap]:
        data = self._data
        batch_size = len(data)

        predict_from_scratch = [
            self._adapt_out(
                self._task_predict_fn(
                    self._adapt_in(data[0])
                )
            )
        ]

        u.validate(predict_from_scratch, "predict_from_scratch", {
            "angles": tf.TensorSpec(shape=[1, None, None], dtype=u.dtype()),
        })

        predict_from_seed = [
            self._adapt_out(
                self._task_predict_fn(
                    self._adapt_in(d),
                    seed_len = 20,
                )
            )
            for d in data
        ]

        # use tf.nest.map_structure to stack all component tensors
        predict_from_seed = tf.nest.map_structure(
            lambda *xs: tf.concat(xs, axis=0),
            *predict_from_seed,
        )

        u.validate(predict_from_seed, "predict_from_seed", {
            "angles": tf.TensorSpec(shape=[batch_size, None, None], dtype=u.dtype()),
        })

        step_dim = hv.Dimension(("step", "Step"))
        batch_dim = hv.Dimension(("batch", "Batch"))

        def img(data, title):
            data = ein.rearrange(data, "f h j d -> (h j d) f")
            return hv.Raster(data.numpy()).opts(
                title=title,
                cmap='twilight',
                aspect='equal',
            )

        def get_title(name, i_batch=None):
            if tf.is_tensor(name):
                name = name.numpy().decode('utf-8')
            if i_batch is not None:
                return f"{name} {i_batch}"
            else:
                return name

        hmaps = [
            hv.HoloMap(
                {
                    (i_step): img(data, get_title(title))
                },
                kdims=[step_dim],
                label=ident
            )
            for ident, (title, data) in predict_from_scratch.items()
        ]

        hmaps += [
            hv.HoloMap(
                {
                    (i_step, i_batch): img(data[i_batch], get_title(titles[i_batch], i_batch))
                    for i_batch in range(batch_size)
                },
                kdims=[step_dim, batch_dim],
                label=ident
            )
            for ident, (titles, data) in predict_from_seed.items()
        ]

        return hmaps




if __name__ == '__main__':
    import doctest
    doctest.testmod()
