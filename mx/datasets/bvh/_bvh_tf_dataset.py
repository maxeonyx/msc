from dataclasses import dataclass
from os import PathLike
from typing import Any, Callable, Literal, Union

import holoviews as hv
from mx.datasets.utils import make_decimate

from mx.prelude import *
from mx.progress import Progress, create_progress_manager
from mx.tasks import VectorSequenceAngleMSE
from mx import utils as u
from mx.visualizer import HoloMapVisualization, StatefulVisualization, Visualization
from mx.utils import Einshape, DSets
from mx.pipeline import MxDataset, Task
import mx.predict as pred

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
                    "angles": ein.rearrange(x["angles"], f"... f h j d -> ... f (h j d)") if not isinstance(x["angles"], tf.RaggedTensor) else x["angles"].merge_dims(-2, -1).merge_dims(-2, -1),
                    # remove indices for the flattened feature dim, keep only
                    # frame indices
                    "seq_idxs": x["idxs"][..., :, 0, 0, 0, :1],
                    "extra": x["extra"],
                }
            self.adapt_in = adapt_in
            def adapt_out(x):
                # adapt output from a task's predict_fn into dataset-specific format
                return {
                    "angles": ein.rearrange(
                        x["angles"],
                        f"... f (h j d) -> ... f h j d",
                        h=self.n_hands,
                        j=self.n_joints_per_hand,
                        d=self.n_dof_per_joint
                    ),
                }
            self.adapt_out = adapt_out
        else:
            raise NotImplementedError(f"Dataset {type_name(self)} does not support Task {type_name(task)}. If using autoreload in IPython, try restarting the interpreter.")

        assert self.adapt_in is not None, "Forgot to set self.adapt_in"
        assert self.adapt_out is not None, "Forgot to set self.adapt_out"


    def load(self, batch_size, test_batch_size, force_cache_reload: bool=False) -> DSets:

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

        # ragged batch
        dsets = DSets(
            train=dsets.train.apply(tf.data.experimental.dense_to_ragged_batch(batch_size)),
            test=dsets.test.apply(tf.data.experimental.dense_to_ragged_batch(test_batch_size)),
            val=dsets.val.apply(tf.data.experimental.dense_to_ragged_batch(test_batch_size)),
        )

        return dsets

    def get_visualizations(self, viz_batch_size, task_specific_predict_fn) -> dict[str, Visualization]:

        assert self.adapt_in is not None, "Must call dataset.configure(task) before dataset.get_visualizations()"

        viz_data = next(iter(
            self.load(viz_batch_size, viz_batch_size, force_cache_reload=False)
            .test
        ))
        #  # add batch dim, tbc how i'm gonna do batching because i need to support ragged
        #  # sequence dims and einops doesn't like RaggedTensor
        # viz_data = tf.expand_dims(viz_data, axis=0)

        return u.list_to_box([
            BVHImageViz(
                bvh_ds=self,
                data=viz_data,
                task_predict_fn=task_specific_predict_fn,
                adapt_in=self.adapt_in,
            ),
        ])

@export
class BVHImageViz(Visualization):

    def __init__(
        self,
        bvh_ds: BvhDataset,
        data: dict,
        task_predict_fn: Callable[[dict[str, tf.Tensor]], pred.PredictOutputs],
        adapt_in: Callable,
        name = "bvh_imgs",
        desc = "BVH Image Viz",
        output_dir=None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            output_dir=output_dir,
        )

        self.bvh_ds = bvh_ds

        # validate data in "dataset" format
        u.validate(data, "data",  {
            "angles": tf.RaggedTensorSpec(
                shape=[None, None, bvh_ds.n_hands, bvh_ds.n_joints_per_hand, bvh_ds.n_dof_per_joint],
                dtype=u.dtype(),
                ragged_rank=1,
            ),
            "idxs": tf.RaggedTensorSpec(
                shape=[None, None, bvh_ds.n_hands, bvh_ds.n_joints_per_hand, bvh_ds.n_dof_per_joint, 4],
                dtype=tf.int32,
                ragged_rank=1,
            ),
            "extra": {
                "orig_angles": tf.RaggedTensorSpec(
                    shape=[None, None, bvh_ds.n_hands, bvh_ds.n_joints_per_hand, bvh_ds.n_dof_per_joint],
                    dtype=u.dtype(),
                    ragged_rank=1,
                ),
                "filename": tf.RaggedTensorSpec(shape=[None], dtype=tf.string),
            },
        })
        self._data = data
        self._task_predict_fn = task_predict_fn
        self._adapt_in = adapt_in

    def render(self, timestep, output_dir: PathLike=None, pm: Progress=None) -> list[hv.HoloMap]:
        data = self._data

        out = self._task_predict_fn(
            self._adapt_in(data),
            pm=pm,
        )

        out.inputs_imgs=ein.rearrange(
            out.inputs_imgs,
            '... (h j d chan) -> ... (h j d) chan',
            h=self.bvh_ds.n_hands,
            j=self.bvh_ds.n_joints_per_hand,
            d=self.bvh_ds.n_dof_per_joint,
            chan=3,
        )
        out.sampling_order_imgs=ein.rearrange(
            out.sampling_order_imgs,
            '... chan -> ... () chan',
        )
        out.mean_anims=ein.rearrange(
            out.mean_anims,
            '... (h j d chan) -> ... (h j d) chan',
            h=self.bvh_ds.n_hands,
            j=self.bvh_ds.n_joints_per_hand,
            d=self.bvh_ds.n_dof_per_joint,
            chan=3,
        )

        pred.plot_noninteractive(self.output_location(output_dir) / f"plot_{timestep}.png", [out])

    def _get_uri(self, output_dir) -> str:
        return (self.output_location(output_dir) / "plot.png").absolute().as_uri()



@export
class BVHHolomapsViz(HoloMapVisualization):

    def __init__(self,
        bvh_ds: BvhDataset,
        data: list[dict[str, tf.Tensor | dict[str, tf.Tensor]]],
        task_predict_fn: Callable[[dict[str, tf.Tensor]], dict[str, tf.Tensor]],
        adapt_in: Callable,
        name = "bvh_hmap",
        desc = "BVH HMap Viz",
        output_dir=None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            output_dir=output_dir,
        )

        # validate data in "dataset" format
        u.validate(data, "data", {
            "angles": tf.RaggedTensorSpec(shape=[None, None, 2, 17, 3], dtype=u.dtype(), ragged_rank=1),
            "idxs": tf.RaggedTensorSpec(shape=[None, None, 2, 17, 3, 4], dtype=tf.int32, ragged_rank=1),
            "extra": {
                "orig_angles": tf.RaggedTensorSpec(shape=[None, None, 2, 17, 3], dtype=u.dtype(), ragged_rank=1),
                "filename": tf.TensorSpec(shape=[None], dtype=tf.string),
            },
        })
        self.ds = bvh_ds
        self._data = data
        self._task_predict_fn = task_predict_fn
        self._adapt_in = adapt_in

    def _make_hmaps(self, i_step, pm:Progress=None) -> list[hv.HoloMap]:
        data = self._data
        n_seeds = len(data)


        predict_from_scratch = self._task_predict_fn(
            self._adapt_in(data),
            seed_len=0,
            pm=pm,
        )

        predict_from_seed = self._task_predict_fn(
            self._adapt_in(data),
            seed_len = 20,
            pm=pm,
        )

        n_steps = predict_from_seed.

        tp(predict_from_scratch)
        tp(predict_from_seed)

        step_dim = hv.Dimension(("step", "Step"))
        seed_dim = hv.Dimension(("seed", "Seed"))

        def img(data):
            data = ein.rearrange(data, "f (h j d c) -> (h j d) f c", h=self.ds.n_hands, j=self.ds.n_joints_per_hand, d=self.ds.n_dof_per_joint, c=3)
            return hv.RGB(data.numpy()).opts(
                aspect='equal',
            )

        hmaps = [
            hv.HoloMap(
                {
                    (i_seed): img(predict_from_scratch.inputs_imgs[i_seed])
                    for i_seed in range(n_seeds)
                },
                kdims=[seed_dim],
                label="Inputs",
            ),
            hv.HoloMap(
                {
                    (i_seed): hv.RGB(predict_from_scratch.sampling_order_imgs[i_seed, 0, :, None, :]).opts(
                        aspect='equal',
                    )
                    for i_seed in range(n_seeds)
                },
                kdims=[seed_dim],
                label="Sampling Order",
            ),
        ]

        hmaps += [
            hv.HoloMap(
                {
                    (i_step, i_seed): img(predict_from_seed.mean_anims[i_seed, i_step])
                    for i_seed in range(n_seeds)
                    for i_step in range(n_steps)
                },
                kdims=[step_dim, seed_dim],
                label="Mean Anim"
            )
        ]

        return hmaps



if __name__ == '__main__':

    from mx.predict import predict, PredictInputs

    ds = BvhDataset()

    def output_fn(angles, dist):
        angles = u.angle_wrap(angles)
        angles = angles[..., None]
        colors = u.colorize(angles, vmin=-pi, vmax=pi, cmap="twilight_shifted")
        return ein.rearrange(
            colors,
            '... feat chan -> ... (feat chan)',
        )

    viz_batch_size = 5
    seed_len = 30
    output_len = 1000
    def predict_fn(inputs, seed_len=seed_len, output_len=output_len, pm=None):
        u.validate(inputs, "inputs", {
            "angles": tf.TensorSpec(shape=[viz_batch_size, None, ds.n_hands, ds.n_joints_per_hand, ds.n_dof_per_joint], dtype=u.dtype()),
            "idxs": tf.TensorSpec(shape=[viz_batch_size, None, ds.n_hands, ds.n_joints_per_hand, ds.n_dof_per_joint, 4], dtype=tf.int32),
        })

        return predict(
            name="Inputs",
            desc="Showing inputs",
            cfg=PredictInputs(
                out_seq_shape=[output_len],
                out_feat_shape=[ds.n_hands * ds.n_joints_per_hand * ds.n_dof_per_joint * 3],
                model=None,
                input_data=ein.rearrange(
                    inputs["angles"][:, :seed_len],
                    "b t h j d -> b t (h j d)",
                ),
                idxs=inputs["idxs"][:, :output_len, 0, 0, 0, :1],
            ),
            pm=pm,
            data_out_fn=output_fn,
            default_out_var_val=ein.repeat(
                pred.DEFAULT_BG_COLOR,
                'chan -> (h j d chan)',
                h=ds.n_hands,
                j=ds.n_joints_per_hand,
                d=ds.n_dof_per_joint,
            )
        )
    data = next(iter(
        ds.load(viz_batch_size, viz_batch_size).test
    ))
    viz = BVHImageViz(
        bvh_ds=ds,
        data=data,
        task_predict_fn=predict_fn,
        adapt_in=lambda x: {
            "angles": tf.gather(x["angles"], tf.range(output_len), axis=1),
            "idxs": tf.gather(x["idxs"], tf.range(output_len), axis=1),
        }
    )
    hviz = BVHHolomapsViz(
        bvh_ds=ds,
        data=data,
        task_predict_fn=predict_fn,
        adapt_in=lambda x: {
            "angles": tf.gather(x["angles"], tf.range(output_len), axis=1),
            "idxs": tf.gather(x["idxs"], tf.range(output_len), axis=1),
        }
    )

    with create_progress_manager() as pm:
        hviz(pm=pm)
        viz(pm=pm)
