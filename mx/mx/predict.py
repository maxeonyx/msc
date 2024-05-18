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

from contextlib import ExitStack
from os import PathLike
import threading
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from matplotlib.figure import Figure

from mx.progress import Progress, create_progress_manager
from mx.prelude import *


@export
class PredictInputs:
    """
    If the model supports querying, predicting gets the mean of the remaining outputs at each step.
    """
    def __init__(
        self,
        out_seq_shape: list[int] | tf.Tensor | tf.TensorShape,
        raw_feat_shape: list[int] | tf.Tensor | tf.TensorShape,
        viz_feat_shape: list[int] | tf.Tensor | tf.TensorShape,
        model: Model,
        model_supports_querying: bool | Literal["yes_and_use"] = False,
        model_outputs_distribution: bool | int = False,
        sampling_order: Literal["fixed", "highest_entropy", "lowest_entropy"] = "fixed",
        seed_data: tf.Tensor | None = None,
        target_data: tf.Tensor | None = None,
        target_data_idxs: tf.Tensor | None = None,
        idxs: tf.Tensor | None = None,
    ):
        self.out_seq_shape = out_seq_shape
        self.raw_feat_shape = raw_feat_shape
        self.viz_feat_shape = viz_feat_shape
        self.model = model
        self.model_supports_querying = model_supports_querying
        self.model_outputs_distribution = model_outputs_distribution
        self.sampling_order = sampling_order

        if sampling_order == "fixed":
            assert idxs is not None, "If sampling_order is 'fixed', idxs must be provided"

        if seed_data is not None:
            assert idxs is not None, "If seed_data is provided, idxs must be provided"
        self.seed_data = seed_data

        self.idxs = idxs

        if target_data is not None:
            assert target_data_idxs is not None, "If target_data is provided, target_data_idxs must be provided"
        self.target_data = target_data
        self.target_data_idxs = target_data_idxs

@export
class PredictOutputs(Box):
    def __init__(
        self,
        seed_data_viz: tf.Tensor | None,
        target_data_viz: tf.Tensor | None,
        sampling_order_viz: tf.Tensor,
        mean_raw_anims: tf.Tensor,
        mean_viz_anims: tf.Tensor,
        mean_entropy_anims: tf.Tensor | None,
        samples_anims: tf.Tensor | None,
        samples_viz_anims: tf.Tensor | None,
        samples_entropy_anims: tf.Tensor | None,
    ):
        self.seed_data_viz = seed_data_viz
        self.target_data_viz = target_data_viz
        self.sampling_order_viz = sampling_order_viz
        self.mean_raw_anims = mean_raw_anims
        self.mean_viz_anims = mean_viz_anims
        self.mean_entropy_anims = mean_entropy_anims
        self.samples_anims = samples_anims
        self.samples_viz_anims = samples_viz_anims
        self.samples_entropy_anims = samples_entropy_anims

    def numpy(self) -> PredictOutputs:

        if self.seed_data_viz is not None and tf.is_tensor(self.seed_data_viz):
            self.seed_data_viz = self.seed_data_viz.numpy()
        if self.target_data_viz is not None and tf.is_tensor(self.target_data_viz):
            self.target_data_viz = self.target_data_viz.numpy()
        if tf.is_tensor(self.sampling_order_viz):
            self.sampling_order_viz = self.sampling_order_viz.numpy()
        if tf.is_tensor(self.mean_raw_anims):
            self.mean_raw_anims = self.mean_raw_anims.numpy()
        if tf.is_tensor(self.mean_viz_anims):
            self.mean_viz_anims = self.mean_viz_anims.numpy()
        if self.mean_entropy_anims is not None and tf.is_tensor(self.mean_entropy_anims):
            self.mean_entropy_anims = self.mean_entropy_anims.numpy()
        if self.samples_anims is not None and tf.is_tensor(self.samples_anims):
            self.samples_anims = self.samples_anims.numpy()
        if self.samples_viz_anims is not None and tf.is_tensor(self.samples_viz_anims):
            self.samples_viz_anims = self.samples_viz_anims.numpy()
        if self.samples_entropy_anims is not None and tf.is_tensor(self.samples_entropy_anims):
            self.samples_entropy_anims = self.samples_entropy_anims.numpy()

        return self

@u.tf_function
def predict_core(
    model,
    i_step,
    start_at,
    break_var,
    sample_fns,
    viz_fns,
    vals_var: tf.Variable,
    idxs_var: tf.Variable,
    viz_var: tf.Variable,
    query_all: bool,
    sampling_order: Literal["fixed", "highest_entropy", "lowest_entropy"],
):
    """
    Run a batch of auto-regressive predictions, and write the results to `out_var`.
    """

    n_seed_inps = vals_var.shape[0]
    n_sample_fns = vals_var.shape[1]
    seq_len = vals_var.shape[2]
    n_feature_dims = vals_var.shape[3]

    assert idxs_var.shape[0] == n_seed_inps, f"idxs_var.shape[0]={idxs_var.shape[0]}  ≠  n_seed_inps={n_seed_inps}"
    if sampling_order == "fixed":
        assert idxs_var.shape[1] == 1
    else:
        assert idxs_var.shape[1] == n_sample_fns, f"idxs_var.shape[1]={idxs_var.shape[1]}  ≠  n_sample_fns={n_sample_fns}"
    assert idxs_var.shape[2] == seq_len, f"idxs_var.shape[2]={idxs_var.shape[2]}  ≠  seq_len={seq_len}"
    n_indices = idxs_var.shape[-1]

    n_steps = viz_var.shape[0]
    assert viz_var.shape[1] == n_seed_inps, f"out_var.shape[1]={viz_var.shape[1]}  ≠  n_seed_inps={n_seed_inps}"
    assert viz_var.shape[2] == n_sample_fns, f"out_var.shape[2]={viz_var.shape[2]}  ≠  n_sample_fns={n_sample_fns}"
    n_output_fns = viz_var.shape[3]
    # <n_indices> dims here
    n_output_channels = viz_var.shape[-1]

    assert shape(vals_var)[2] == seq_len, f"input variable must be long enough to receive the output, got shape(inp_vals)=={shape(vals_var)}, seq_len={seq_len}"
    assert len(viz_fns) == n_output_fns, f"out_fns must be the same length as out_var.shape[3], but got len(out_fns)={len(viz_fns)} and n_output_fns={n_output_fns}"

    assert len(sample_fns) == n_sample_fns, f"sample_fns must be the same length as inp_vals.shape[1], but got len(sample_fns)={len(sample_fns)} and n_sample_fns={n_sample_fns}"

    if query_all:
        raise NotImplementedError("query_all not implemented")

    if sampling_order != "fixed":
        raise NotImplementedError("dynamic sampling order not implemented yet")

    if model is None:
        n_steps = n_steps - 1

    i_step.assign(0)
    while i_step < n_steps and not break_var:

        i = start_at + i_step

        if model is None:

            n_future_steps = (n_steps+1)-(i_step+1)
            # scatter this input into the output at all future steps
            scatter_idxs = u.multidim_indices_range(
                tf.range(i+1, n_steps+1),
                (0, n_seed_inps),
                (0, n_sample_fns),
                (0, n_output_fns),
            )
            scatter_idxs = tf.concat([
                tf.tile(
                    scatter_idxs[:, :, :, :, :],
                    [            1, 1, 1, 1, 1],
                ),
                tf.tile(
                    idxs_var[None,   :, :, None,         i, :],
                    [n_future_steps, 1, 1, n_output_fns,    1],
                )
            ], axis=-1)
            scatter_idxs = ein.rearrange(
                scatter_idxs,
                'step seed sample out idx -> (step seed sample out) idx',
            )

            [out_fn] = viz_fns
            viz_var.scatter_nd_update(
                scatter_idxs,
                out_fn(ein.rearrange(
                    tf.tile(
                        vals_var[None,   :, :, None,         i, :],
                        [n_future_steps, 1, 1, n_output_fns,    1],
                    ),
                    'step seed sample out idx -> (step seed sample out) idx',
                ), None),
                name="scatter_inps_to_outs",
            )

        else: # model is not None

            # flatten into batch_size before model, then unflatten after

            context = {
                    "context/values": ein.rearrange(
                        vals_var[:, :, :i, :],
                        'seed sample seq ... -> (seed sample) seq ...',
                    ),
                }

            inputs = {
                **context,
                "context/inp_idxs": ein.rearrange(
                    idxs_var[:, :, :i, :],
                    'seed sample seq idx -> (seed sample) seq idx',
                ),
                "context/tar_idxs": ein.rearrange(
                    idxs_var[:, :, :i+1, :],
                    'seed sample seq idx -> (seed sample) seq idx',
                ),
            }
            # ic(inputs)
            outputs = model(inputs, training=False)

            # tp(outputs, "outputs")
            # tf.print(outputs)

            # todo implement a transformer that runs incrementally
            output = outputs[:, -1]

            if isinstance(output, tfd.Distribution):
                output = tfd.BatchReshape(output, batch_shape=[n_seed_inps, n_sample_fns])
            else:
                output = ein.rearrange(
                    output,
                    '(seed sample) ... -> seed sample ...',
                    seed=n_seed_inps,
                    sample=n_sample_fns,
                )

            # zip outputs with out_fns along the batch dimension
            for i_samp, s_fn in enumerate(sample_fns):
                out = output[:, i_samp]
                samp_out = s_fn(out)

                vals_var[:, i_samp, i].assign(samp_out)

                for i_out, o_fn in enumerate(viz_fns):
                    outp = o_fn(samp_out, out)

                    # write new output to out_var
                    n_future_steps = n_steps-i_step
                    scatter_idxs = u.multidim_indices_range(
                        tf.range(i_step, n_steps),
                        (0, n_seed_inps),
                        i_samp,
                        i_out,
                    )
                    scatter_idxs = tf.concat([
                        scatter_idxs,
                        tf.tile(
                            idxs_var[None,   :, i_samp:i_samp+1, None, i, :],
                            [n_future_steps, 1, 1,               1,       1],
                        ),
                    ], axis=-1)
                    scatter_idxs = ein.rearrange(
                        scatter_idxs,
                        'step seed samp out idx -> (step seed samp out) idx',
                    )

                    outp = tf.tile(
                        outp[None, :],
                        [n_future_steps, 1] + [1]*(outp.shape.rank-1),
                    )

                    # tp(out_var, "out_var")
                    # tp(scatter_idxs, "scatter_idxs")
                    # tp(outp, "outp")

                    viz_var.scatter_nd_update(
                        scatter_idxs,
                        ein.rearrange(
                            outp,
                            'step seed ... -> (step seed) ...',
                        ),
                        name="scatter_new_outs_to_outs",
                    )

        i_step.assign_add(1)

DEFAULT_BG_COLOR = tf.constant([255, 100, 150], tf.uint8)
DEFAULT_ALT_BG_COLOR = tf.constant([200, 255, 180], tf.uint8)


def predict(
    name: str,
    desc: str,
    cfg: PredictInputs,
    pm: Progress | None | Literal[True],
    default_viz_val: tf.Tensor = DEFAULT_BG_COLOR,
    default_order_out_var_val: tf.Tensor = DEFAULT_ALT_BG_COLOR,
    raw_out_fn = None,
    viz_out_fn = None,
    outp_to_inp_fn = None,
) -> PredictOutputs:
    """
    Run `predict_core` on another thread to support quick keyboard interrupt & live progress.
    Create variables for the outputs.
    """

    def default_data_out_fn(samp, dist):
        return u.colorize(samp, cmap='gray')

    if viz_out_fn is None:
        viz_out_fn = default_data_out_fn
        assert cfg.viz_feat_shape[-1] == 3, f"if viz_out_fn is not provided, viz_feat_shape must have 3 channels. Got viz_feat_shape={cfg.viz_feat_shape}"

    raw_out_fn = raw_out_fn or (lambda x: x)

    def entropy_out_fn(samp, dist):
        return u.colorize(dist.entropy(), vmin=0, vmax=None, cmap='viridis')

    out_seq_shape = cfg.out_seq_shape
    output_len = prod(out_seq_shape)
    n_indices = len(out_seq_shape)

    seq_dim_names = [f"out_seq_{i}" for i in range(len(cfg.out_seq_shape))]
    seq_dims = { name: size for name, size in zip(seq_dim_names, cfg.out_seq_shape) }
    seq_ein_spec = " ".join(seq_dim_names)

    viz_feat_shape = cfg.viz_feat_shape
    viz_feat_dim_names = [f"viz_feat_{i}" for i in range(len(viz_feat_shape))]
    viz_feat_dims = { name: size for name, size in zip(viz_feat_dim_names, viz_feat_shape) }
    viz_feat_ein_spec = " ".join(viz_feat_dim_names)
    viz_ein_spec = f"{seq_ein_spec} {viz_feat_ein_spec}"
    viz_dims = { **seq_dims, **viz_feat_dims }

    raw_feat_shape = cfg.raw_feat_shape
    raw_feat_dim_names = [f"raw_feat_{i}" for i in range(len(raw_feat_shape))]
    raw_feat_dims = { name: size for name, size in zip(raw_feat_dim_names, raw_feat_shape) }
    raw_feat_ein_spec = " ".join(raw_feat_dim_names)
    raw_ein_spec = f"{seq_ein_spec} {raw_feat_ein_spec}"
    raw_dims = { **seq_dims, **raw_feat_dims }


    def order_to_rgb(idxs):
        idxs = u.multidim_idxs_to_flat_idxs(idxs, shape=out_seq_shape)
        return u.colorize(idxs, vmin=0, vmax=output_len, cmap='plasma')

    if cfg.model_outputs_distribution:
        is_distribution = True
        n_samples = cfg.model_outputs_distribution
    else:
        is_distribution = False

    assert cfg.idxs.shape.rank in [2, 3], f"idxs must be a flat sequence of multi-dimensional indices, or a batch of sequences of multi-dimensional indices. Got shape(cfg.idxs)={cfg.idxs.shape}"
    if cfg.seed_data is not None or cfg.idxs.shape.rank == 3:
        is_seeded = True

        has_seed_data = cfg.seed_data is not None

    else:
        is_seeded = False

    if cfg.sampling_order != "fixed":
        is_dynamic = True
    else:
        is_dynamic = False

    if cfg.target_data is not None:
        target_data_len = cfg.target_data.shape[1]

    # todo: refactoring this to be a bunch of exclusive cases
    if is_seeded:
        # seed provided
        inp_data_idxs = cfg.idxs
        inp_data = cfg.seed_data

        n_seed_inps = shape(inp_data_idxs)[0]

        if has_seed_data:
            assert shape(inp_data)[0] == n_seed_inps, f"n_seed_inps (dim 0) of inp_data must match n_seed_inps, got shape(inp_data)={shape(inp_data)} and n_seed_inps={n_seed_inps}"
            seed_data_len = shape(inp_data)[1]

        # if dynamic, idxs same length as seed
        # otherwise idxs is full seq length
        if has_seed_data and is_dynamic:
            assert shape(inp_data_idxs)[1] == seed_data_len, f"seq_len (dim 1) of inp_data_idxs must match seed_data_len, got shape(inp_data_idxs)={shape(inp_data_idxs)} and seed_data_len={seed_data_len}"
        else:
            assert shape(inp_data_idxs)[1] == output_len, f"seq_len (dim 1) of inp_data_idxs must match output_len, got shape(inp_data_idxs)={shape(inp_data_idxs)} and output_len={output_len}"

        assert shape(inp_data_idxs)[-1] == n_indices, f"n_indices (dim -1) of inp_data_idxs must match n_indices, got shape(inp_data_idxs)={shape(inp_data_idxs)} and n_indices={n_indices}"
    elif not is_seeded and not is_dynamic:
        # fixed sampling order, no seed
        # full seq of idxs required
        inp_data_idxs = cfg.idxs
        assert shape(inp_data_idxs)[0] == output_len, f"seq_len (dim 0) of inp_data_idxs must match output_len, got shape(inp_data_idxs)={shape(inp_data_idxs)} and output_len={output_len}"
        assert shape(inp_data_idxs)[-1] == n_indices, f"n_indices (dim -1) of inp_data_idxs must match n_indices, got shape(inp_data_idxs)={shape(inp_data_idxs)} and n_indices={n_indices}"
    elif not is_seeded and is_dynamic:
        # dynamic sampling order, no seed
        # not even idxs required
        pass
    else:
        raise ValueError(f"Invalid cfg: {cfg}")

    # ic(
    #     n_seed_inps if is_seeded else None,
    #     n_samples if is_distribution else None,
    #     seed_len if is_seeded else None,
    #     output_len,
    #     n_indices
    # )



    if cfg.model is None:

        assert is_seeded and has_seed_data, "If no model is provided, seed data must be provided"

        model = cfg.model
        n_steps = seed_data_len + 1
        viz_fns = [viz_out_fn]
        n_viz_fns = len(viz_fns)
        n_feature_dims = inp_data.shape[-1]

        # always start at 0 even if seeded
        start_at = 0

    else:

        model = cfg.model
        # dbg(model.input_shape, "model.input_shape")
        # dbg(model.name, "model name")
        if 'context/values' in model.input_shape:
            n_feature_dims = model.input_shape["context/values"][-1]
        elif 'context/tokens' in model.input_shape:
            n_feature_dims = 0

        if is_seeded and has_seed_data:
            start_at = seed_data_len
            n_steps = output_len - seed_data_len
        else:
            start_at = 0
            n_steps = output_len

        if is_distribution:
            sample_fns = [lambda dist: dist.mean()]
            sample_fns += [lambda dist: dist.sample()] * n_samples
            del n_samples
            n_sample_fns = len(sample_fns)
            viz_fns = [viz_out_fn, entropy_out_fn]
            n_viz_fns = len(viz_fns)
        else:
            viz_fns = [viz_out_fn]
            n_viz_fns = len(viz_fns)

    ############# create input variables #############

    viz_var = tf.Variable(
        initial_value=ein.repeat(
            default_viz_val,
            f'{viz_feat_ein_spec} -> steps seed sample outfns {viz_ein_spec}',
            steps=n_steps,
            seed=n_seed_inps if is_seeded else 1, # if not seeded, still need to have a seed dim
            sample=n_sample_fns if is_distribution else 1, # if not distribution, still need to have a sample dim
            outfns=n_viz_fns,
            **viz_dims,
        ),
        trainable=False,
        name="out_vals",
    )
    vals_var = tf.Variable(
        initial_value=tf.zeros([
            n_seed_inps if is_seeded else 1, # if not seeded, still need to have a seed dim
            n_sample_fns if is_distribution else 1, # if not distribution, still need to have a sample dim
            output_len,
            n_feature_dims
        ], u.dtype()),
        trainable=False,
        name="inp_vals",
    )
    idxs_var = tf.Variable(
        initial_value=tf.zeros([
            n_seed_inps if is_seeded else 1, # if not seeded, still need to have a seed dim
            n_sample_fns if is_distribution else 1, # if not distribution, still need to have a sample dim
            output_len,
            n_indices,
        ], tf.int32),
        trainable=False,
        name="idxs_var",
    )

    if is_seeded and has_seed_data:
        vals_var[:, :, :seed_data_len].assign(ein.repeat(
            inp_data,
            'seed seq feat -> seed sample seq feat',
            sample=n_sample_fns if is_distribution else 1,
        ))

    if is_dynamic and is_seeded:
        idxs_var[:, :, :seed_data_len].assign(ein.repeat(
            inp_data_idxs,
            'seed seq idx -> seed sample seq idx',
            sample=n_sample_fns if is_distribution else 1,
        ))
    elif is_seeded:
        idxs_var.assign(ein.repeat(
            inp_data_idxs,
            'seed seq idx -> seed sample seq idx',
            sample=n_sample_fns if is_distribution else 1,
        ))
    elif is_dynamic:
        pass # no idxs required
    else:
        idxs_var.assign(ein.repeat(
            inp_data_idxs,
            'seq idx -> seed sample seq idx',
            seed=n_seed_inps if is_seeded else 1,
            sample=n_sample_fns if is_distribution else 1,
        ))


    ########## call predict_core ##########

    with ExitStack() as stack:
        if pm is True:
            pm = stack.enter_context(create_progress_manager())

        if pm is not None:
            sub_pm, progbar = stack.enter_context(pm.enter_progbar(total=n_steps, name=name, desc=desc, delete_on_success=True))

        step_var = tf.Variable(0, dtype=tf.int32, trainable=False, name="step_var")
        break_var = tf.Variable(False, dtype=tf.bool, trainable=False, name="break_var")
        exc = None
        def run_predict_in_thread():
            nonlocal exc
            try:
                predict_core(
                    model=model,
                    start_at=start_at,
                    i_step=step_var,
                    break_var=break_var,
                    viz_var=viz_var,
                    vals_var=vals_var,
                    idxs_var=idxs_var,
                    sample_fns=sample_fns if is_distribution else ([outp_to_inp_fn] if outp_to_inp_fn is not None else [lambda x: x]),
                    viz_fns=viz_fns,
                    sampling_order=cfg.sampling_order if is_dynamic else "fixed",
                    query_all=False,
                )
            except Exception as e:
                exc = e
                raise

        thread = threading.Thread(target=run_predict_in_thread)

        thread.start()
        try:
            while thread.is_alive():
                if pm is not None:
                    progbar.count = step_var.value().numpy()
                    progbar.refresh()
                time.sleep(0.1)
        except KeyboardInterrupt:
            break_var.assign(True)
            thread.join()
            raise
        finally:
            thread.join()
            if exc is not None:
                raise exc

    ######### make image of input data #########
    bg_img = ein.repeat(
        default_viz_val,
        f'{viz_feat_ein_spec} -> seed {viz_ein_spec}',
        seed=n_seed_inps if is_seeded else 1, # if not seeded, still need to have a seed dim
        **viz_dims,
    )

    if is_seeded and has_seed_data:
        seed_data_len
        scatter_data_idxs = tf.concat([
            # seed idxs
            ein.repeat(
                u.multidim_indices_range(
                    tf.range(n_seed_inps if is_seeded else 1),
                ),
                'seed idx -> (seed seq) idx',
                seq=seed_data_len,
            ),
            # seq idxs
            ein.rearrange(
                inp_data_idxs[:, :seed_data_len],
                'seed seq idx -> (seed seq) idx',
            ),
        ], axis=-1)
        seed_data_viz = tf.tensor_scatter_nd_update(
            bg_img,
            scatter_data_idxs,
            viz_out_fn(ein.rearrange(
                inp_data,
                'seed seq chan -> (seed seq) chan',
            ), None),
            name="scatter_inputs_img",
        )
    else:
        seed_data_viz = None

    if cfg.target_data is not None:
        scatter_data_idxs = tf.concat([
            # seed idxs
            ein.repeat(
                u.multidim_indices_range(
                    tf.range(n_seed_inps if is_seeded else 1),
                ),
                'seed idx -> (seed seq) idx',
                seq=target_data_len,
            ),
            # seq idxs
            ein.rearrange(
                cfg.target_data_idxs[:, :target_data_len],
                'seed seq idx -> (seed seq) idx',
            ),
        ], axis=-1)
        target_data_viz = tf.tensor_scatter_nd_update(
            bg_img,
            scatter_data_idxs,
            viz_out_fn(ein.rearrange(
                cfg.target_data,
                'seed seq chan -> (seed seq) chan',
            ), None),
            name="scatter_targets_viz",
        )
    else:
        target_data_viz = None

    ######### make image of sampling order #########

    bg_img = ein.repeat(
        default_order_out_var_val,
        f'chan -> seed sample {seq_ein_spec} chan',
        seed=n_seed_inps if is_seeded else 1,
        sample=n_sample_fns if is_distribution else 1,
        **seq_dims,
    )
    scatter_order_idxs = tf.concat([
        ein.repeat(
            u.multidim_indices_range(
                tf.range(n_seed_inps if is_seeded else 1),
                tf.range(n_sample_fns if is_distribution else 1),
            ),
            'seed sample idx -> (seed sample seq) idx',
            seq=output_len,
        ),
        ein.rearrange(
            idxs_var,
            'seed sample seq idx -> (seed sample seq) idx',
        ),
    ], axis=-1)
    sampling_order_viz = tf.tensor_scatter_nd_update(
        bg_img,
        scatter_order_idxs,
        order_to_rgb(ein.repeat(
            u.multidim_indices(out_seq_shape),
            'seq idx -> (seed sample seq) idx',
            seed=n_seed_inps if is_seeded else 1,
            sample=n_sample_fns if is_distribution else 1,
        )),
    )

    mean_raw_anims = tf.scatter_nd(
        indices=tf.concat([
            ein.repeat(
                u.multidim_indices_range(
                    tf.range(n_seed_inps if is_seeded else 1),
                ),
                'seed idx -> (seed seq) idx',
                seq=output_len,
            ),
            ein.rearrange(
                idxs_var[:, 0],
                'seed seq idx -> (seed seq) idx',
            ),
        ], axis=-1),
        updates=raw_out_fn(ein.rearrange(
            vals_var[:, 0],
            f'seed seq ... -> (seed seq) ...',
        )),
        shape=[n_seed_inps, *out_seq_shape, *raw_feat_shape],
    )
    mean_viz_anims = ein.rearrange(
        viz_var[:, :, 0, 0],
        'step seed ... -> seed step ...',
    )
    if is_distribution:
        mean_entropy_anims = ein.rearrange(
            viz_var[:, :, 0, 1],
            'step seed ... -> seed step ...',
        )
        samples_raw_anims = tf.scatter_nd(
            indices=tf.concat([
                u.multidim_indices_range(
                    tf.range(n_seed_inps if is_seeded else 1),
                    tf.range(1, n_sample_fns),
                ),
                ein.rearrange(
                    idxs_var[:, 1:],
                    'seed samp seq idx -> (seed samp seq) idx',
                ),
            ], axis=-1),
            updates=ein.rearrange(
                vals_var[:, 1:],
                f'seed samp seq ({raw_feat_ein_spec}) -> (seed samp seq) {raw_feat_ein_spec}',
                **raw_feat_dims,
            ),
            shape=[n_seed_inps, n_sample_fns-1, *out_seq_shape, *raw_feat_shape],
        )
        samples_viz_anims = ein.rearrange(
            viz_var[:, :, 1:, 0],
            'step seed sample ... -> seed sample step ...',
        )
        samples_entropy_anims = ein.rearrange(
            viz_var[:, :, 1:, 1],
            'step seed sample ... -> seed sample step ...',
        )
    else:
        mean_entropy_anims = None
        samples_raw_anims = None
        samples_viz_anims = None
        samples_entropy_anims = None
    # all the viz types with non-dynamic output
    return PredictOutputs(
        seed_data_viz=seed_data_viz,
        target_data_viz=target_data_viz,
        sampling_order_viz=sampling_order_viz,
        mean_raw_anims=mean_raw_anims,
        mean_viz_anims=mean_viz_anims,
        mean_entropy_anims=mean_entropy_anims,
        samples_anims=samples_raw_anims,
        samples_viz_anims=samples_viz_anims,
        samples_entropy_anims=samples_entropy_anims,
    )

def imgs(out: PredictOutputs, fig: Figure):
    cfg = Box()

    cfg.out = out

    cfg.n_seeds = out.mean_raw_anims.shape[0]
    cfg.n_samples = out.samples_anims.shape[1] if out.samples_anims is not None else 1
    s = cfg.n_samples - 1
    cfg.n_steps = out.mean_raw_anims.shape[1]
    t = cfg.n_steps - 1


    cfg.fig = fig
    cfg.ax = cfg.fig.subplots(3, cfg.n_seeds)

    if len(cfg.ax.shape) == 1:
        cfg.ax = cfg.ax[:, None]

    cfg.input_imgs = []
    for i in range(cfg.n_seeds):
        cfg.input_imgs.append(cfg.ax[0, i].imshow(out.seed_data_viz[i]))

    cfg.sampling_order_imgs = []
    for i in range(cfg.n_seeds):
        cfg.sampling_order_imgs.append(cfg.ax[1, i].imshow(out.sampling_order_viz[i, s]))

    cfg.mean_anim_imgs = []
    for i in range(cfg.n_seeds):
        cfg.mean_anim_imgs.append(cfg.ax[2, i].imshow(out.mean_raw_anims[i, t]))

    return cfg

def plot(outs: list[PredictOutputs], block=False):
    """
    Plot the outputs of a prediction, then show to the screen for interactive
    viewing.
    """

    plt.ion()

    fig = plt.figure()

    figs = fig.subfigures(len(outs) + 1, height_ratios=[0.1] + [1] * len(outs))

    print("mean_anims_shape", outs[0].mean_raw_anims.shape)

    max_n_steps = max(out.mean_raw_anims.shape[1] for out in outs)
    max_n_samples = max(
        (
            out.samples_anims.shape[2] if outs[0].samples_anims is not None else 1
        )
        for out in outs
    )

    cfgs = [
        imgs(out, figs[i + 1]) for i, out in enumerate(outs)
    ]

    def update(_):
        t = int(time_slider.val)
        s = int(samples_slider.val)
        for cfg in cfgs:
            this_t = min(t, cfg.n_steps - 1)
            this_s = min(s, cfg.n_samples - 1)
            for i in range(cfg.n_seeds):
                cfg.mean_anim_imgs[i].set_data(cfg.out.mean_anims[i, this_t])
        for fig in figs:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    time_slider_ax = figs[0].add_axes([0.2, 0., 0.6, 0.3])
    time_slider = plt.Slider(time_slider_ax, label="Time", valmin=0, valmax=max_n_steps - 1, valinit=max_n_steps -1)
    time_slider.on_changed(update)

    samples_slider_ax = figs[0].add_axes([0.2, 0.5, 0.6, 0.3])
    samples_slider = plt.Slider(samples_slider_ax, label="Sample #", valmin=0, valmax=max_n_samples - 1, valinit=0)
    samples_slider.on_changed(update)

    plt.show(block=block)

def plot_noninteractive(file: PathLike, outs: list[PredictOutputs], select_steps: list[int] = None):
    """
    Plot the outputs of a prediction, then save to disk.
    """

    fig = plt.figure()

    figs = fig.subfigures(len(outs) + 1, height_ratios=[0.1] + [1] * len(outs))

    cfgs = [
        imgs(out, figs[i + 1]) for i, out in enumerate(outs)
    ]

    plt.savefig(file)
    plt.close()

# if __name__ == "__main__":

#     n_seeds = 5
#     examples = predict(
#         name="Examples",
#         desc="Making examples animation...",
#         cfg=PredictInputs(
#             model=None,
#             out_seq_shape=[10, 10],
#             out_feat_shape=[3],
#             seed_data=tf.random.uniform([n_seeds, 10*10, 1]),
#             idxs=tf.tile(
#                 u.multidim_indices([10, 10], elide_rank_1=False)[None],
#                 [n_seeds, 1, 1]
#             ),
#         ),
#         pm=True,
#     )

#     model_inputs = u.input_dict(
#         Input([None, 1], dtype=u.dtype(), name="context/values"),
#         Input([None, 2], dtype=tf.int32,  name="context/inp_idxs"),
#     )

#     def demo_model(inputs):
#         v = inputs["context/values"]
#         batch_size = tf.shape(v)[0]
#         seq_len = tf.shape(v)[1]
#         n_idxs = tf.shape(v)[2]
#         v = tf.concat([
#             tf.ones([batch_size, 1, n_idxs], dtype=tf.float32),
#             v,
#         ], axis=1)
#         return tf.reduce_mean(v, axis=1)[:, None, :] * 0.9


#     model = Model(
#         inputs=model_inputs,
#         outputs=demo_model(model_inputs),
#         name="test_model",
#     )

#     predictions = predict(
#         name="Predict",
#         desc="Making predictions animation...",
#         cfg=PredictInputs(
#             out_seq_shape=[10, 10],
#             out_feat_shape=[3],
#             idxs=u.multidim_indices([10, 10]),
#             model=model,
#             model_supports_querying=False,
#             model_outputs_distribution=False,
#         ),
#         pm=True,
#     )

#     rand_idxs = tf.stack([
#         tf.random.shuffle(u.multidim_indices([10, 10]))for _ in range(n_seeds)
#     ], axis=0)

#     rand_predictions = predict(
#         name="RandPredict",
#         desc="Making random predictions animation...",
#         cfg=PredictInputs(
#             out_seq_shape=[10, 10],
#             out_feat_shape=[3],
#             idxs=rand_idxs,
#             model=model,
#             model_supports_querying=False,
#             model_outputs_distribution=False,
#         ),
#         pm=True,
#     )

#     # d = Dense(1)
#     # dist_layer = tfp.layers.DistributionLambda(lambda mean: tfp.distributions.Normal(mean, 1.0))
#     # def demo_distribution_model(inputs):
#     #     v = inputs["context/values"]
#     #     return dist_layer(d(v))

#     # distribution_model = Model(
#     #     inputs=model_inputs,
#     #     outputs=demo_distribution_model(model_inputs),
#     #     name="test_distribution_model",
#     # )

#     # distribution_predictions = predict(
#     #     name="Distribution Predict",
#     #     desc="Making distribution predictions animation...",
#     #     cfg=PredictInputs(
#     #         out_seq_shape=[10, 10],
#     #         idxs=u.multidim_indices([10, 10]),
#     #         model=distribution_model,
#     #         model_supports_querying=False,
#     #         model_outputs_distribution=4, # 4 samples
#     #     ),
#     #     pm=True,
#     # )


#     plot([examples, predictions, rand_predictions], block=True)
