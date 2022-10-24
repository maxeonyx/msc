"""
Produce a bunch of stuff that can be visualized.

N = the number of sequences to sample, if sampling.


If the model output is a distribution, we get multiple samples rather than just one prediction,
and we also get to look at the entropy.

If the model supports querying over the entire sequence at once, we get previews of the mean,
entropy etc. as it samples.

A "Video" contains:
    a. An image of the "seed" inputs
    c. An image of the sampling order

    (If a `model` is provided)
    b. The predicted outputs at each step

    (If the model supports querying)
    c. The mean of the remaining outputs at each step

    (If the model output is a distribution)
    d. The entropy of the remaining outputs at each step

#### What this function returns ####


(If `inputs` is provided)
1.  A video of each of the examples being taken bit-by-bit.

(If the model supports querying)
3.  A video of each of the examples being taken bit-by-bit, in random order.

(If `inputs` and `seed_len` are provided)
4.  "Seeded" predictions: Predict each output position given the previous output positions
    starting with a batch of seed inputs.

    a. One video of the mean/modal sequence

    (If the model output is a distribution)
    b. N videos of sampled sequences

5.  "Unseeded" predictions: Predict each output position given the previous output positions,
    starting with only the begin token.

    a. One video of the mean/modal sequence

    (If the model output is a distribution)
    b. N videos of sampled sequences

(If the model output is a distribution & supports querying)
5. "Dynamic order" predictions: Predict each output position given the previous output positions,
    starting with only the begin token, but with the order of the output positions chosen based
    on some statistic of the distribution.

    a. Highest-entropy-first
        a. One video of the mean/modal sequence
        b. N videos of sampled sequences
    b. Lowest-entropy-first
        a. One video of the mean/modal sequence
        b. N videos of sampled sequences
"""

from __future__ import annotations

import abc
from ast import Mod
from cProfile import label
from contextlib import ExitStack
from email.policy import default
from multiprocessing import context
from typing import Generic, overload
from mx.progress import Progress, create_progress_manager
from mx.prelude import *
from mx.utils import dtype

class PredictInputs(Box, abc.ABC):
    def __init__(
        self,
        out_seq_shape: list[int] | tf.Tensor | tf.TensorShape,
        out_feat_shape: list[int] | tf.Tensor | tf.TensorShape,
        sampling_order: Literal["fixed", "highest_entropy", "lowest_entropy"] = "fixed"
    ):
        self.out_seq_shape = out_seq_shape
        self.out_feat_shape = out_feat_shape
        self.sampling_order = sampling_order

class PredictOutputs(Box, abc.ABC):
    def __init__(
        self,
        inputs_img: tf.Tensor,
        sampling_order_img: tf.Tensor
    ):
        self.inputs_img = inputs_img
        self.sampling_order_img = sampling_order_img

class InputOnly_Inputs(PredictInputs):
    "No model, only data and a sampling order"
    def __init__(
        self,
        out_seq_shape: list[int] | tf.Tensor | tf.TensorShape,
        out_feat_shape: list[int] | tf.Tensor | tf.TensorShape,
        input_data: tf.Tensor,
        idxs: tf.Tensor
    ):
        super().__init__(out_seq_shape, out_feat_shape)
        self.input_data = input_data
        self.idxs = idxs

class InputOnly_Outputs(PredictOutputs):
    def __init__(
        self,
        inputs_img: tf.Tensor,
        sampling_order_img: tf.Tensor,
        inputs_anim: tf.Tensor
    ):
        super().__init__(inputs_img, sampling_order_img)
        self.inputs_anim = inputs_anim

class PredictWithModel_Inputs(PredictInputs, abc.ABC):
    """
    If the model supports querying, predicting gets the mean of the remaining outputs at each step.
    """
    def __init__(
        self,
        out_seq_shape: list[int] | tf.Tensor | tf.TensorShape,
        out_feat_shape: list[int] | tf.Tensor | tf.TensorShape,
        model: Model,
        model_supports_querying: bool | Literal["yes_and_use"],
        model_outputs_distribution: bool | int,
        sampling_order: Literal["fixed", "highest_entropy", "lowest_entropy"] = "fixed"
    ):
        super().__init__(out_seq_shape, out_feat_shape, sampling_order)
        self.model = model
        self.model_supports_querying = model_supports_querying
        self.model_outputs_distribution = model_outputs_distribution

class PredictWithModel_Outputs(PredictOutputs):
    def __init__(
        self,
        inputs_img: tf.Tensor,
        sampling_order_img: tf.Tensor,
        mean_anim: tf.Tensor,
        samples_anim: tf.Tensor | None,
        entropy_anim: tf.Tensor | None,
    ):
        super().__init__(inputs_img, sampling_order_img)
        self.mean_anim = mean_anim
        self.samples_anim = samples_anim
        self.entropy_anim = entropy_anim

class FromScratch_Inputs(PredictWithModel_Inputs):
    """
    Predict each output position given the previous output positions, starting with only the begin token.
    If the model supports querying, predicting gets the mean of the remaining outputs at each step.
    If the model output is a distribution, we get multiple samples rather than just one prediction, and we also get to look at the entropy.
    """
    def __init__(
        self,
        out_seq_shape: list[int] | tf.Tensor | tf.TensorShape,
        out_feat_shape: list[int] | tf.Tensor | tf.TensorShape,
        model: Model,
        model_supports_querying: bool | Literal["yes_and_use"],
        model_outputs_distribution: bool | int,
        idxs: tf.Tensor
    ):
        super().__init__(
            out_seq_shape,
            out_feat_shape,
            model,
            model_supports_querying,
            model_outputs_distribution
        )
        self.idxs = idxs


class FromScratch_Outputs(PredictWithModel_Outputs):
    def __init__(
        self,
        inputs_img: tf.Tensor,
        sampling_order_img: tf.Tensor,
        mean_anim: tf.Tensor,
        samples_anim: tf.Tensor | None,
        entropy_anim: tf.Tensor | None,
    ):
        super().__init__(inputs_img, sampling_order_img, mean_anim, samples_anim, entropy_anim)


# class FromSeed_Inputs(PredictWithModel_Inputs):
#     "Predict with seed inputs"
#     seed_data: tf.Tensor
#     idxs: tf.Tensor

# class DynamicOrderFromScratch(PredictWithModel_Inputs):
#     order_type: Literal["highest_entropy", "lowest_entropy"]

# class DynamicOrderFromSeed(PredictWithModel_Inputs):
#     order_type: Literal["highest_entropy", "lowest_entropy"]
#     seed_data: tf.Tensor
#     seed_idxs: tf.Tensor


@dataclass
class Video:
    seed_inputs: tf.Tensor
    sampling_order: tf.Tensor
    mean_prediction: tf.Tensor | None
    mean_entropies: tf.Tensor | None
    sampled_predictions: tf.Tensor | None
    sampled_entropies: tf.Tensor | None

# @u.tf_function
def predict_core(
    model,
    i_step,
    start_at,
    break_var,
    query_all,
    sample_fns,
    out_fns,
    inp_vals: tf.Variable,
    inp_idxs: tf.Variable,
    sampling_order: Literal["fixed", "highest_entropy", "lowest_entropy"],
    out_var: tf.Variable,
):
    """
    Run a batch of auto-regressive predictions, and write the results to `out_var`.
    """

    ic(inp_vals)
    ic(inp_idxs)

    n_seed_inps = inp_vals.shape[0]
    n_sample_fns = inp_vals.shape[1]
    seq_len = inp_vals.shape[2]
    n_feature_dims = inp_vals.shape[3]

    assert inp_idxs.shape[0] == n_seed_inps, f"inp_idxs.shape[0]={inp_idxs.shape[0]}  ≠  n_seed_inps={n_seed_inps}"
    if sampling_order == "fixed":
        assert inp_idxs.shape[1] == seq_len, f"inp_idxs.shape[1]={inp_idxs.shape[1]}  ≠  seq_len={seq_len}"
    else:
        assert inp_idxs.shape[1] == n_sample_fns, f"inp_idxs.shape[1]={inp_idxs.shape[1]}  ≠  n_sample_fns={n_sample_fns}"
        assert inp_idxs.shape[2] == seq_len, f"inp_idxs.shape[2]={inp_idxs.shape[2]}  ≠  seq_len={seq_len}"
    n_indices = inp_idxs.shape[-1]

    n_steps = out_var.shape[0]
    assert out_var.shape[1] == n_seed_inps, f"out_var.shape[1]={out_var.shape[1]}  ≠  n_seed_inps={n_seed_inps}"
    assert out_var.shape[2] == n_sample_fns, f"out_var.shape[2]={out_var.shape[2]}  ≠  n_sample_fns={n_sample_fns}"
    n_output_fns = out_var.shape[3]
    # n_indices dims
    n_output_channels = out_var.shape[-1]

    assert shape(inp_vals)[2] == seq_len, f"input variable must be long enough to receive the output, got shape(inp_vals)=={shape(inp_vals)}, seq_len={seq_len}"
    assert len(out_fns) == n_output_fns, f"out_fns must be the same length as out_var.shape[3], but got len(out_fns)={len(out_fns)} and n_output_fns={n_output_fns}"
    assert len(sample_fns) == n_sample_fns, f"sample_fns must be the same length as inp_vals.shape[1], but got len(sample_fns)={len(sample_fns)} and n_sample_fns={n_sample_fns}"

    if query_all is not None:
        raise NotImplementedError("query_all not implemented")

    if sampling_order != "fixed":
        raise NotImplementedError("dynamic sampling order not implemented yet")

    i_step.assign(0)
    while i_step < n_steps and not break_var:

        i = start_at + i_step

        if model is None:

            step_idxs = tf.tile(
                i_step[None, None, None, None, None, None],
                [1, n_seed_inps, n_sample_fns, n_output_fns, i, 1],
            )
            seed_input_idxs = tf.tile(
                tf.range(n_seed_inps)[None, :, None, None, None, None],
                [1, 1, n_sample_fns, n_output_fns, i, 1],
            )
            sample_fn_idxs = tf.tile(
                tf.range(n_sample_fns)[None, None, :, None, None, None],
                [1, n_seed_inps, 1, n_output_fns, i, 1],
            )
            data_idxs = tf.tile(
                inp_idxs[None, :, None, None, :i, :],
                [1, n_seed_inps, n_sample_fns, n_output_fns, 1, 1],
            )
            out_fn_idxs = tf.tile(
                tf.range(n_output_fns)[None, None, None, :, None, None],
                [1, n_seed_inps, n_sample_fns, 1, i, 1],
            )
            scatter_idxs = ein.rearrange(
                tf.concat([step_idxs, seed_input_idxs, sample_fn_idxs, out_fn_idxs, data_idxs], axis=-1),
                'step seed sample out data idx -> (step seed sample out data) idx',
            )

            [out_fn] = out_fns
            out_var.scatter_nd_update(
                scatter_idxs,
                out_fn(ein.rearrange(
                    inp_vals[:, :, :i, :],
                    'seed sample seq ... -> (seed sample seq) ...',
                ), None),
                name="scatter_inps_to_outs",
            )

        else: # model is not None


            # flatten into batch_size before model, then unflatten after
            ctx_vals = ein.rearrange(
                inp_vals[:, :, :i, :],
                'seed sample seq ... -> (seed sample) seq ...',
            )
            if sampling_order == "fixed":
                ctx_idxs = ein.rearrange(
                    inp_idxs[:, :i, :],
                    'seed seq idx -> seed seq idx',
                )
            else:
                ctx_idxs = ein.rearrange(
                    inp_idxs[:, :, :i, :],
                    'seed sample seq idx -> (seed sample) seq idx',
                )

            inputs = {
                "context/values": ctx_vals[:, :i],
                "context/inp_idxs": ctx_idxs[:, :i],
                "context/tar_idxs": ctx_idxs[:, :i+1],
                # "query/values": inp_var[:, i],
                # "query/inp_idxs": inp_var[:, i],
            }
            ic(inputs)
            outputs = model(inputs, training=False)

            # unflatten into seed, sample, out_fn, step, idx
            outputs = ein.rearrange(
                outputs,
                '(seed sample) seq ... -> sample seed seq ...',
                seed=n_seed_inps,
                sample=n_sample_fns,
            )

            # todo implement a transformer that runs incrementally
            output = outputs[:, -1, ...]
            ic(outputs)
            # zip outputs with out_fns along the batch dimension
            for i_samp, (out, s_fn) in enumerate(zip(output, sample_fns)):
                ic(out)
                samp_out = s_fn(out)
                inp_vals[:, i_samp, i].assign(samp_out)

                outs = tf.stack([
                    out_fn(samp_out, out)
                    for out_fn in out_fns
                ], axis=0)


                step_idxs = tf.tile(
                    i_step[None, None, None, None, None, None],
                    [1, n_seed_inps, n_sample_fns, n_output_fns, 1, 1],
                )
                seed_input_idxs = tf.tile(
                    tf.range(n_seed_inps)[None, :, None, None, None, None],
                    [1, 1, n_sample_fns, n_output_fns, 1, 1],
                )
                sample_fn_idxs = tf.tile(
                    tf.range(n_sample_fns)[None, None, :, None, None, None],
                    [1, n_seed_inps, 1, n_output_fns, 1, 1],
                )
                data_idxs = tf.tile(
                    inp_idxs[None, :, None, None, i:i+1, :],
                    [1, n_seed_inps, n_sample_fns, n_output_fns, 1, 1],
                )
                out_fn_idxs = tf.tile(
                    tf.range(n_output_fns)[None, None, None, :, None, None],
                    [1, n_seed_inps, n_sample_fns, 1, 1, 1],
                )
                ic([step_idxs, seed_input_idxs, sample_fn_idxs, out_fn_idxs, data_idxs])
                scatter_idxs = ein.rearrange(
                    tf.concat([step_idxs, seed_input_idxs, sample_fn_idxs, out_fn_idxs, data_idxs], axis=-1),
                    'step seed sample out data idx -> (step seed sample out data) idx',
                )
                ic(scatter_idxs)
                out_var.scatter_nd_update(
                    scatter_idxs,
                    ein.rearrange(
                        outs,
                        'seed outfns ... -> (seed outfns) ...',
                    ),
                    name="scatter_outs_to_outs",
                )

        i_step.assign_add(1)

    # tp(out_var, "out_var")
    # tf.print(out_var)

DEFAULT_BG_COLOR = tf.constant([255, 100, 150], tf.uint8)

def predict(
    name: str,
    desc: str,
    cfg: Union[InputOnly_Inputs, FromScratch_Inputs],
    pm: Progress | None | Literal[True],
    default_out_var_val: tf.Tensor = DEFAULT_BG_COLOR,
    data_out_fn = None,
) -> Union[InputOnly_Outputs, FromScratch_Outputs]:
    """
    Run `predict_core` on another thread to support quick keyboard interrupt & live progress.
    Create variables for the outputs.
    """

    def default_data_out_fn(samp, dist):
        return u.colorize(samp, cmap='gray')
    data_out_fn = data_out_fn or default_data_out_fn

    def entropy_out_fn(samp, dist):
        return u.colorize(dist.entropy(), cmap='viridis')

    output_shape = cfg.out_seq_shape
    output_len = prod(output_shape)

    def order_to_rgb(idxs):
        idxs = u.multidim_idxs_to_flat_idxs(idxs, shape=output_shape)
        return u.colorize(idxs, vmin=0, vmax=output_len, cmap='plasma')


    # todo: refactoring this to be a bunch of exclusive cases
    if cfg.sampling_order == "fixed" and hasattr(cfg, "input_data") and hasattr(cfg, "idxs"):
        inp_data_idxs = cfg.idxs
        n_seed_inps = shape(inp_data_idxs)[0]
        inp_data = cfg.input_data
        seed_len = shape(inp_data)[1]
        assert shape(inp_data)[0] == shape(inp_data_idxs)[0], f"n_seed_inps of inp_data and inp_data_idxs must match, got shape(inp_data)={shape(inp_data)} and shape(inp_data_idxs)={shape(inp_data_idxs)}"
        n_samples = shape(inp_data)[1]
    elif hasattr(cfg, 'idxs'):
        inp_data_idxs = cfg.idxs
        n_samples = shape(inp_data_idxs)[1]
    else:
        assert seed_len == 0
        inp_data_idxs = tf.zeros([n_seed_inps, seed_len, len(output_shape)], tf.int32)

    if hasattr(cfg, "input_data"):

        if cfg.sampling_order != "fixed":
            assert shape(inp_data)[0] == shape(inp_data_idxs)[0], f"n_seed_inps of inp_data and inp_data_idxs must match, got shape(inp_data)={shape(inp_data)} and shape(inp_data_idxs)={shape(inp_data_idxs)}"
            assert shape(inp_data)[1] == shape(inp_data_idxs)[1], f" if `dynamic_order` is True, then the second dimension of inp_data and inp_data_idxs must match, got shape(inp_data)={shape(inp_data)} and shape(inp_data_idxs)={shape(inp_data_idxs)}"


    n_indices = shape(inp_data_idxs)[-1]

    out_dim_names = [f"out_{i}" for i in range(len(output_shape))]
    out_dims = { name: size for name, size in zip(out_dim_names, output_shape) }
    out_ein_spec = " ".join(out_dim_names)

    if isinstance(cfg, InputOnly_Inputs):

        model = None

        n_steps = output_len + 1
        sample_fns = [lambda x: x] * n_seed_inps
        n_sample_fns = len(sample_fns)
        out_fns = [data_out_fn]
        n_out_fns = len(out_fns)

        n_feature_dims = inp_data.shape[-1]
        start_at = 0

    elif isinstance(cfg, PredictWithModel_Inputs):

        model = cfg.model
        n_feature_dims = model.input_shape["context/values"][-1]
        start_at = seed_len

        if cfg.model_outputs_distribution is False:
            sample_fns = [lambda x: x]
            n_sample_fns = len(sample_fns)
            out_fns = [data_out_fn]
            n_out_fns = len(out_fns)
        else:
            sample_fns = [lambda dist: dist.mean()] + [lambda dist: dist.sample()] * cfg.model_outputs_distribution
            n_sample_fns = len(sample_fns)
            out_fns = [data_out_fn, entropy_out_fn]
            n_out_fns = len(out_fns)


        if isinstance(cfg, FromScratch_Inputs):
            inp_data = tf.zeros([n_seed_inps, 0, n_feature_dims], u.dtype())
            n_steps = output_len
        else:
            n_steps = output_len - seed_len

    else:
        raise NotImplementedError(f"config type {type_name(cfg)} not implemented yet")

    out_var = tf.Variable(
        initial_value=ein.repeat(
            default_out_var_val,
            f'chan -> steps seed sample out {out_ein_spec} chan',
            steps=n_steps,
            seed=n_seed_inps,
            sample=n_sample_fns,
            out=n_out_fns,
            **out_dims,
        ),
        trainable=False,
        name="out_vals",
    )
    inp_vals = tf.Variable(
        initial_value=tf.zeros([n_seed_inps, n_sample_fns, output_len, n_feature_dims], u.dtype()),
        trainable=False,
        name="inp_vals",
    )
    inp_vals[:, :, :seed_len].assign(
        ein.repeat(
            inp_data,
            'seed seq feat -> seed sample seq feat',
            sample=n_sample_fns,
        )
    )
    if cfg.sampling_order == "fixed":
        inp_idxs = tf.Variable(
            initial_value=tf.zeros([n_seed_inps, output_len, n_indices], tf.int32),
            trainable=False,
            name="inp_idxs",
        )
        inp_idxs[:, :].assign(inp_data_idxs)
    else:
        inp_idxs = tf.Variable(
            initial_value=tf.zeros([n_seed_inps, n_sample_fns, output_len, n_indices], tf.int32),
            trainable=False,
            name="inp_idxs",
        )
        inp_idxs[:, :, :seed_len].assign(
            ein.repeat(
                inp_data_idxs,
                'seed seq idx -> seed sample seq idx',
                sample=n_sample_fns,
            )
        )

    bg_img = ein.repeat(
        default_out_var_val,
        f'chan -> seed {out_ein_spec} chan',
        seed=n_seed_inps,
        **out_dims,
    )

    scatter_data_idxs = tf.concat([
        # seed_inp idxs
        ein.repeat(
            tf.range(n_seed_inps),
            'seed -> (seed seq) ()',
            seq=seed_len,
        ),
        # seq and out idxs
        ein.rearrange(
            inp_data_idxs[:, :seed_len],
            'seed seq idx -> (seed seq) idx',
        ),
    ], axis=-1)
    inputs_img = tf.tensor_scatter_nd_update(
        bg_img,
        scatter_data_idxs,
        data_out_fn(ein.rearrange(
            inp_data,
            'seed seq chan -> (seed seq) chan',
        ), None),
        name="scatter_inputs_img",
    )

    with ExitStack() as stack:
        if pm is True:
            pm = stack.enter_context(create_progress_manager(u.get_run_name()))

        if pm is not None:
            stack.enter_context(pm.enter_progbar(total=n_steps, name=name, desc=desc, delete_on_success=True))

        step_var = tf.Variable(0, dtype=tf.int32, trainable=False, name="step_var")
        break_var = tf.Variable(False, dtype=tf.bool, trainable=False, name="break_var")

        predict_core(
            model=model,
            start_at=start_at,
            i_step=step_var,
            break_var=break_var,
            out_var=out_var,
            inp_vals=inp_vals,
            inp_idxs=inp_idxs,
            sampling_order=cfg.sampling_order,
            sample_fns=sample_fns,
            out_fns=out_fns,
            query_all=None,
        )

    # all the viz types with non-dynamic output
    if cfg.sampling_order == "fixed":

        scatter_order_idxs = tf.concat([
            ein.repeat(
                tf.range(n_seed_inps),
                'seed -> (seed seq) ()',
                seq=output_len,
            ),
            ein.rearrange(
                inp_idxs,
                'seed seq idx -> (seed seq) idx',
            ),
        ], axis=-1)
        sampling_order_img = tf.tensor_scatter_nd_update(
            bg_img,
            scatter_order_idxs,
            order_to_rgb(ein.rearrange(
                inp_idxs,
                'seed seq idx -> (seed seq) idx',
            )),
        )
    else:
        # idxs has extra dim `samp`
        raise NotImplementedError("dynamic order not implemented yet")

    if isinstance(cfg, InputOnly_Inputs):
        return InputOnly_Outputs(
            inputs_img=inputs_img,
            sampling_order_img=sampling_order_img,
            inputs_anim=out_var,
        )
    elif isinstance(cfg, FromScratch_Inputs):
        return FromScratch_Outputs(
            inputs_img=inputs_img,
            sampling_order_img=sampling_order_img,
            mean_anim=out_var,
            samples_anim=None,
            entropy_anim=None,
        )
    else:
        raise NotImplementedError(f"config type {type_name(cfg)} not implemented yet")


if __name__ == "__main__":

    import holoviews as hv
    hv.extension('bokeh')

    # no model - just output a video of the inputs being taken bit-by-bit

    outs = predict(
        name="Examples",
        desc="Making examples animation...",
        cfg=InputOnly_Inputs(
            out_seq_shape=[10, 10],
            out_feat_shape=[3],
            input_data=tf.random.uniform([1, 10*10, 1]),
            idxs=u.multidim_indices([10, 10], elide_rank_1=False)[None, ...],
        ),
        pm=True,
    )

    outs.inputs_anim = ein.rearrange(
        outs.inputs_anim,
        'step seed samp outfn h w c -> (seed samp outfn) step h w c',
    ).numpy()
    outs.inputs_img = ein.rearrange(
        outs.inputs_img,
        'b h w c -> b h w c',
    ).numpy()
    outs.sampling_order_img = ein.rearrange(
        outs.sampling_order_img,
        'b h w c -> b h w c',
    ).numpy()

    n_seeds = shape(outs.inputs_anim)[0]
    seeds_dim = hv.Dimension(('seed', 'Seed'), range=(0, n_seeds))
    n_time = shape(outs.inputs_anim)[1]
    time_dim = hv.Dimension(('t', 'Time'), range=(0, n_time))

    input_data_grid = hv.GridSpace(
        {
            i_seed: hv.RGB(outs.inputs_img[i_seed])
            for i_seed in range(n_seeds)
        },
        kdims=[seeds_dim],
        label="Input Data"
    ).opts(
        sizing_mode="scale_both",
    )
    sampling_order_grid = hv.GridSpace(
        {
            i_seed: hv.RGB(outs.sampling_order_img[i_seed], cmap='plasma')
            for i_seed in range(n_seeds)
        },
        kdims=[seeds_dim],
        label="Sampling Order"
    ).opts(
        sizing_mode="scale_both",
    )

    select_steps = [0, 1, 2, 3, 24, 45, 76, 97, 98, 99, 100]
    inputs_anim_img = hv.GridSpace(
        {
            i_seed: hv.HoloMap(
                {
                    i_step: hv.RGB(outs.inputs_anim[i_seed, i_step])
                    for i_step in select_steps
                },
                kdims=[time_dim],
            )
            for i_seed in range(n_seeds)
        },
        kdims=[seeds_dim],
        label="Inputs Anim",
    )

    layout = hv.Layout([
        input_data_grid,
        sampling_order_grid,
        inputs_anim_img,
    ]).cols(1).opts(
        sizing_mode="stretch_both",
    )

    hv.save(layout, "test_predict.examples.ignore.html")

    # open in web browser
    import webbrowser
    webbrowser.open("file://" + os.path.abspath("test_predict.examples.ignore.html"))


    model_inputs = u.input_dict(
        Input([None, 1], dtype=u.dtype(), name="context/values"),
        Input([None, 1], dtype=tf.int32,  name="context/inp_idxs"),
        # Input([None, 1], dtype=tf.int32,  name="context/tar_idxs"),

        # Input([None, 1], dtype=u.dtype(), name="query/values"),
        # Input([None, 1], dtype=tf.int32,  name="query/inp_idxs"),
        # Input([None, 1], dtype=tf.int32,  name="query/tar_idxs"),
    )

    def demo_model(inputs):
        v = inputs["context/values"]
        v = tf.concat([
            tf.constant(1., dtype=u.dtype(), shape=[1, 1, 1]),
            v,
        ], axis=1)
        return tf.reduce_mean(v, axis=1)[:, None, :]
        # batch_size = shape(inputs["context/values"])[0]
        # feature_dim = shape(inputs["context/values"])[2]
        # return tf.reduce_mean(
        #     tf.concat([
        #         inputs["context/values"],
        #         # inputs["query/values"],
        #         tf.random.uniform([batch_size, 1, feature_dim]),
        #     ], axis=1),
        #     axis=-1,
        # )

    model = Model(
        inputs=model_inputs,
        outputs=demo_model(model_inputs),
        name="test_model",
    )
    idxs = ein.repeat(
        u.multidim_indices([10, 10], elide_rank_1=False),
        'seq idxs -> seed seq idxs',
        seed=5,
    )
    outs = predict(
        name="Predict",
        desc="Making predictions animation...",
        cfg=FromScratch_Inputs(
            out_seq_shape=[10, 10],
            out_feat_shape=[3],
            idxs=idxs,
            model=model,
            model_supports_querying=False,
            model_outputs_distribution=False,
        ),
        pm=True,
    )

    outs.mean_anim = ein.rearrange(
        outs.mean_anim,
        'step seed samp outfn h w c -> (samp outfn seed) step h w c',
    ).numpy()
    input_data_grid = ein.rearrange(
        outs.inputs_img,
        'seed h w c -> seed h w c',
    ).numpy()
    sampling_order_grid = ein.rearrange(
        outs.sampling_order_img,
        'seed h w c -> seed h w c',
    ).numpy()

    n_seeds = shape(outs.mean_anim)[0]
    seeds_dim = hv.Dimension(('seed', 'Seed'), range=(0, n_seeds))
    n_time = shape(outs.mean_anim)[1]
    time_dim = hv.Dimension(('t', 'Time'), range=(0, shape(outs.mean_anim)[0]))

    input_data_grid = hv.GridSpace(
        {
            i_seed: hv.RGB(input_data_grid[i_seed])
            for i_seed in range(shape(outs.inputs_img)[0])
        },
        kdims=[seeds_dim],
        label="Input Data"
    ).opts(
        sizing_mode="scale_both",
    )
    sampling_order_grid = hv.GridSpace(
        {
            b: hv.RGB(sampling_order_grid[b], cmap='plasma')
            for b in range(shape(outs.sampling_order_img)[0])
        },
        kdims=[seeds_dim],
        label="Sampling Order"
    ).opts(
        sizing_mode="scale_both",
    )

    select_steps = [0, 1, 2, 3, 24, 45, 76, 97, 98, 99]
    mean_anim_img = hv.GridSpace(
        {
            i_seed: hv.HoloMap(
                {
                    i_step: hv.RGB(outs.mean_anim[i_seed, i_step])
                    for i_step in select_steps
                },
                kdims=[time_dim],
            )
            for i_seed in range(n_seeds)
        },
        kdims=[seeds_dim],
        label="Predictions Anim",
    )

    layout = hv.Layout([
        input_data_grid,
        sampling_order_grid,
        mean_anim_img,
    ]).cols(1).opts(
        sizing_mode="stretch_both",
    )

    hv.save(layout, "test_predict.from_scratch.ignore.html")

    # open in web browser
    import webbrowser
    webbrowser.open("file://" + os.path.abspath("test_predict.from_scratch.ignore.html"))
