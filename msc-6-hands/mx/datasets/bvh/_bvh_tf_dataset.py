from dataclasses import dataclass
from os import PathLike
import re
from typing import Any, Callable, Literal, Union

import holoviews as hv
hv.extension('bokeh')
from mx.datasets.utils import make_decimate

from mx.prelude import *
from mx.progress import Progress, create_progress_manager
from mx.tasks import ForwardAngleAMSE, MultidimTask_DatasetConfig
from mx import utils as u
from mx.visualizer import Visualization
from mx.utils import Einshape, DSets
from mx.pipeline import MxDataset, Task, Task_DatasetConfig
from mx.predict import PredictOutputs, PredictInputs, predict, DEFAULT_BG_COLOR

from . import _bvh

@export
@dataclass
class BvhDataset(MxDataset):
    """
    Config for the BVH dataset pipeline.
    """

    def __init__(
        self,
        do_recluster: bool = False,
        do_decimate: Union[Literal[False], float] = False,
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
        self.do_recluster = do_recluster
        """Whether to recluster the angles to have a circular mean of 0."""
        self.do_decimate = do_decimate
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

        if task.ds_config_cls == Task_DatasetConfig:
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
        elif task.ds_config_cls == MultidimTask_DatasetConfig:
            task.recieve_dataset_config(task.ds_config_cls(
                seq_dims=[None, 2, 17, 3],
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

        circular_means = u.circular_mean(all_angles, axis=0)
        self.circular_means = circular_means

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

        if self.do_recluster:
            dset = map_angles(dset, lambda a: u.recluster(a, circular_means))

        dset = dset.map(lambda x: { **x, "idxs": u.multidim_indices_of(x["angles"], flatten=False) })
        index_einshape = angles_einshape.append_feature_dim("i", angles_einshape.rank)

        if self.do_decimate:
            decimate = make_decimate(self.do_decimate, angles_einshape, other_params=[(index_einshape, tf.int32)])
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

    def recluster(self, angles):

        flat = False
        if angles.shape[-1] == 102:
            flat = True
            angles = ein.rearrange(
                angles,
                f"... (h j d) -> ... h j d",
                h=2,
                j=17,
                d=3,
            )
        assert angles.shape[-3:] == (self.n_hands, self.n_joints_per_hand, self.n_dof_per_joint), f"Expected angles to have shape (..., {self.n_hands}, {self.n_joints_per_hand}, {self.n_dof_per_joint}), got {angles.shape}"
        angles = u.recluster(angles, self.circular_means)
        if flat:
            angles = ein.rearrange(
                angles,
                f"... h j d -> ... (h j d)",
                h=2,
                j=17,
                d=3,
            )
        return angles

    def unrecluster(self, angles):
        assert angles.shape[-3:] == (self.n_hands, self.n_joints_per_hand, self.n_dof_per_joint), f"Expected angles to have shape (..., {self.n_hands}, {self.n_joints_per_hand}, {self.n_dof_per_joint}), got {angles.shape}"
        return u.unrecluster(angles, self.circular_means)

    def get_visualizations(self, model, output_dir) -> dict[str, Visualization]:

        return u.list_to_box([
            BVHHolomapsViz(
                model=model,
                output_dir=output_dir,
            ),
        ])

# @export
# class BVHImageViz(Visualization):

#     def __init__(
#         self,
#         bvh_ds: BvhDataset,
#         data: dict,
#         task_predict_fn: Callable[[dict[str, tf.Tensor]], PredictOutputs],
#         adapt_in: Callable,
#         name = "bvh_imgs",
#         desc = "BVH Image Viz",
#         output_dir=None,
#     ):
#         super().__init__(
#             name=name,
#             desc=desc,
#             output_dir=output_dir,
#         )

#         self.bvh_ds = bvh_ds

#         # validate data in "dataset" format
#         u.validate(data, "data",  {
#             "angles": tf.RaggedTensorSpec(
#                 shape=[None, None, bvh_ds.n_hands, bvh_ds.n_joints_per_hand, bvh_ds.n_dof_per_joint],
#                 dtype=u.dtype(),
#                 ragged_rank=1,
#             ),
#             "idxs": tf.RaggedTensorSpec(
#                 shape=[None, None, bvh_ds.n_hands, bvh_ds.n_joints_per_hand, bvh_ds.n_dof_per_joint, 4],
#                 dtype=tf.int32,
#                 ragged_rank=1,
#             ),
#             "extra": {
#                 "orig_angles": tf.RaggedTensorSpec(
#                     shape=[None, None, bvh_ds.n_hands, bvh_ds.n_joints_per_hand, bvh_ds.n_dof_per_joint],
#                     dtype=u.dtype(),
#                     ragged_rank=1,
#                 ),
#                 "filename": tf.RaggedTensorSpec(shape=[None], dtype=tf.string),
#             },
#         })
#         self._data = data
#         self._task_predict_fn = task_predict_fn
#         self._adapt_in = adapt_in

#     def render(self, timestep, output_dir: PathLike=None, pm: Progress=None) -> list[hv.HoloMap]:
#         data = self._data

#         out = self._task_predict_fn(
#             self._adapt_in(data),
#             pm=pm,
#         )

#         out.inputs_imgs=ein.rearrange(
#             out.inputs_imgs,
#             '... (h j d chan) -> ... (h j d) chan',
#             h=self.bvh_ds.n_hands,
#             j=self.bvh_ds.n_joints_per_hand,
#             d=self.bvh_ds.n_dof_per_joint,
#             chan=3,
#         )
#         out.sampling_order_imgs=ein.rearrange(
#             out.sampling_order_imgs,
#             '... chan -> ... () chan',
#         )
#         out.mean_anims=ein.rearrange(
#             out.mean_anims,
#             '... (h j d chan) -> ... (h j d) chan',
#             h=self.bvh_ds.n_hands,
#             j=self.bvh_ds.n_joints_per_hand,
#             d=self.bvh_ds.n_dof_per_joint,
#             chan=3,
#         )

#         plot_noninteractive(self.output_location(output_dir) / f"plot_{timestep}.png", [out])

#     def _get_uri(self, output_dir) -> str:
#         return (self.output_location(output_dir) / "plot.png").absolute().as_uri()



@export
class BVHHolomapsViz(Visualization):

    def __init__(self,
        model: Model,
        name = "bvh_hmap",
        desc = "BVH HMap Viz",
        output_dir=None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            output_dir=output_dir,
        )

        self.model = model

    def __call__(self, timestep=None, pm: Progress=None, show=True) -> list[hv.HoloMap]:
        model = self.model
        title = self.desc

        if self.model is None:
            inputz = predict_bvh_data(model=None, from_scratch=True, pm=pm)
            plot_bvh_data(inputz, timestep=timestep, title=f"{title} Inputs", show=show, output_dir=self.output_dir)
            return

        from_scratch_output = predict_bvh_data(model, from_scratch=True, pm=pm)
        plot_bvh_data(from_scratch_output, timestep=timestep, title=f"{title} From Scratch", show=show, output_dir=self.output_dir)

        seeded_output = predict_bvh_data(model, from_scratch=False, pm=pm)
        plot_bvh_data(seeded_output, timestep=timestep, title=f"{title} Seeded", show=show, output_dir=self.output_dir)


def raw_out_fn(angles):
    angles = u.angle_wrap(angles)
    return ein.rearrange(
        angles,
        '... (h j d) -> ... h j d',
        h=2,
        j=17,
        d=3,
    )
@export
def predict_bvh_data(model, ds=None, from_scratch=False, targeted=False, pm=True, output_len=500, n_seeds=3, seed_len=30):

    ds = ds or BvhDataset()

    def bvh_to_color(angles, dist):
        angles = u.angle_wrap(angles)
        angles = ds.recluster(angles)
        angles = angles[..., None]
        colors = u.colorize(angles, vmin=-pi, vmax=pi, cmap="twilight_shifted")
        return colors


    inputs = next(iter(ds.load(n_seeds, n_seeds).test))

    inputs = {
        "angles": tf.gather(inputs["angles"], tf.range(output_len), axis=1),
        "idxs": tf.gather(inputs["idxs"], tf.range(output_len), axis=1),
    }
    inputs = {
        "angles": ein.rearrange(
            inputs["angles"],
            "b t h j d -> b t (h j d)",
        ),
        "idxs": inputs["idxs"][:, :, 0, 0, 0, :1],
    }

    if model is None:
        seed_data = inputs["angles"]
        idxs = inputs["idxs"]
        target_data = None
        target_data_idxs = None
    elif from_scratch:
        seed_data = None
        idxs = inputs["idxs"][:1]
        target_data = None
        target_data_idxs = None
    elif targeted:
        target_data = inputs["angles"]
        target_data_idxs = inputs["idxs"]
        seed_data = tf.concat([
            inputs["angles"][:, -1:],
            inputs["angles"][:, :1],
        ], axis=1)
        idxs = tf.concat([
            inputs["idxs"][:, -1:],
            inputs["idxs"][:, :-1],
        ], axis=1)
    else:
        seed_data = inputs["angles"][:, :seed_len]
        idxs = inputs["idxs"]
        target_data = inputs["angles"]
        target_data_idxs = inputs["idxs"]


    dbg({
        "seed_data": seed_data,
        "idxs": idxs,
        "target_data": target_data,
        "target_data_idxs": target_data_idxs,
    }, f"Predicting BVH data with...")

    out = predict(
        name="Predict BVH",
        desc=f"Predicting BVH data...",
        cfg=PredictInputs(
            model=model,
            out_seq_shape=[output_len],
            raw_feat_shape=[2, 17, 3],
            viz_feat_shape=[2*17*3, 3],
            target_data=target_data,
            target_data_idxs=target_data_idxs,
            seed_data=seed_data,
            idxs=idxs,
        ),
        pm=pm,
        outp_to_inp_fn=u.unit_vector_to_angle,
        raw_out_fn=raw_out_fn,
        viz_out_fn=bvh_to_color,
        default_viz_val=ein.repeat(
            DEFAULT_BG_COLOR,
            'chan -> (feat) chan',
            feat=2*17*3,
        ),
    )

    if target_data is not None:
        out.target_raw = ein.rearrange(
            target_data,
            "b t (h j d) -> b t h j d",
            h=2,
            j=17,
            d=3,
        )
    if seed_data is not None:
        out.seed_raw = ein.rearrange(
            seed_data,
            "b t (h j d) -> b t h j d",
            h=2,
            j=17,
            d=3,
        )

    return out

@export
def write_targeted_experiment(output_dir, model=None, data=None, pm=None):

    if model is None:
        assert data is not None
    elif data is None:
        assert model is not None
        data = predict_bvh_data(model, targeted=True, pm=pm)

    # add 90 frames of static position at the end of the animation
    data = tf.concat([
        data,
        ein.repeat(
            data[:, -1], # final frame
            'seed h j d -> seed t h j d',
            t=90, # 90 frames
        ),
    ], axis=1)

    n_seeds = data.mean_raw_anims.shape[0]

    for i in range(n_seeds):
        write_file(data.seed_raw[i, :1], "mean", f"targeted-targ-{i+1}", output_dir)

    for i in range(n_seeds):
        write_file(data.target_raw[i], "mean", f"targeted-ground-truth-{i+1}", output_dir)

    for i in range(n_seeds):
        write_file(data.mean_raw_anims[i], "mean", f"targeted-pred-{i+1}", output_dir)


@export
def plot_bvh_data(out: PredictOutputs, timestep=None, title=None, show=True, output_dir:PathLike=None):

    out = out.numpy()

    title = title or "BVH Data"

    dbg(out, "data to plot")

    n_seeds = out.mean_viz_anims.shape[0]
    n_steps = out.mean_viz_anims.shape[1]

    if timestep is not None:
        trainstep_dim = [hv.Dimension(("train_step", "Training Step"), default=timestep)]
    else:
        trainstep_dim = []
    step_dim = hv.Dimension(("step", "Timestep"), default=n_steps-1)
    seed_dim = hv.Dimension(("seed", "Seed"))

    def img(data, label=None):
        data = ein.rearrange(data, "f (h j d) c -> (h j d) f c", h=2, j=17, d=3, c=3)
        return hv.RGB(data, label=label).opts(
            data_aspect=data.shape[0] / data.shape[1],
            # aspect="equal",
            height=200,
        )

    def key(i_epoch, i_seed, i_step=None):
        if i_seed is not None and i_epoch is not None and i_step is not None:
            return (i_epoch, i_seed, i_step)
        elif i_seed is not None and i_epoch is not None:
            return (i_epoch, i_seed)
        elif i_seed is not None and i_step is not None:
            return (i_seed, i_step)
        elif i_seed is not None:
            return (i_seed,)
    hmaps = []


    if out.target_data_viz is not None:
        hmaps += [
            hv.HoloMap(
                {
                    key(timestep, i_seed): img(out.target_data_viz[i_seed], f"Ground Truth ({i_seed})")
                    for i_seed in range(n_seeds)
                },
                kdims=[*trainstep_dim, seed_dim],
            ).opts(
                responsive=True,
            ),
        ]
    else:
        title += ", (no target data)"

    if out.seed_data_viz is not None:
        hmaps += [
            hv.HoloMap(
                {
                    key(timestep, i_seed): img(out.seed_data_viz[i_seed], f"Seed Inputs ({i_seed})")
                    for i_seed in range(n_seeds)
                },
                kdims=[*trainstep_dim, seed_dim],
            ).opts(
                responsive=True,
            ),
        ]
    else:
        title += ", (no seed data)"

    hmaps += [
        hv.HoloMap(
            {
                key(timestep, i_seed, i_step): img(out.mean_viz_anims[i_seed, i_step], f"Mean Prediction ({i_seed})")
                for i_seed in range(n_seeds)
                for i_step in u.expsteps(n_steps)
            },
            kdims=[*trainstep_dim, seed_dim, step_dim],
            label="Mean Anim"
        ).opts(
            responsive=True,
        ),
    ]
    fig = hv.Layout(hmaps).cols(1).opts(
        shared_axes=False,
        title=title,
        width=1600,
        sizing_mode="stretch_both",
        # responsive=True, # doesn't work
    )

    filename = "bvh"
    if timestep is not None:
        filename += f"_timestep{timestep}"
    if out.seed_data_viz is None:
        filename += "_unseeded"
    filename += ".html"

    output_dir = output_dir or "_bvh_plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    filename = output_dir / filename

    hv.save(fig, filename, widget_location='top', fmt='html', backend='bokeh')

    if show:
        u.show(filename.absolute().as_uri(), title)

@export
def write_file(mean_raw_anim, name: str, type: str, output_dir: PathLike):

    output_dir = output_dir or Path("_bvhfiles")

    output_dir.mkdir(exist_ok=True, parents=True)

    filename = name + "-" + type

    _bvh.write_bvh_files(mean_raw_anim, filename, output_dir=output_dir)


if __name__ == '__main__':

    with create_progress_manager() as pm:
        viz = BVHHolomapsViz(model=None)
        viz(pm=pm)

        model_path = Path('_outputs/blessed/bvh-rand-ranglevec-transformer-tiny-normal-chunklong/AngleCodebookTriples_DecoderOnlyTransformer')
        model = keras.models.load_model(model_path)
        model.load_weights(model_path / 'weights-final')
        viz = BVHHolomapsViz(model=model)
        viz(pm=pm)
