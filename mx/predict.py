from contextlib import ExitStack
from email.policy import default
from mx.progress import Progress, create_progress_manager
from mx.prelude import *
from mx.utils import dtype

@dataclass
class Video:
    seed_inputs: tf.Tensor
    sampling_order: tf.Tensor
    mean_prediction: tf.Tensor | None
    mean_entropies: tf.Tensor | None
    sampled_predictions: tf.Tensor | None
    sampled_entropies: tf.Tensor | None

@u.tf_function
def predict_core(
    model,
    start_at,
    step_var,
    break_var,
    output_len,
    query_all,
    inp_output_fn,
    output_fns,
    inp_var: tf.Variable,
    inp_idxs,
    tar_idxs: tf.Tensor | None | Callable,
    out_var: tf.Variable,
):
    """
    Run a batch of auto-regressive predictions, and write the results to the given `out_vars`.
    """

    assert shape(inp_var)[0] == shape(inp_idxs)[0] == shape(out_var)[0], "batch size must be the same for all inputs"
    if tf.is_tensor(tar_idxs):
        assert shape(inp_var)[0] == shape(tar_idxs)[0], "batch size must be the same for all inputs"
    batch_size = shape(inp_var)[0]

    n_steps = tf.constant(output_len) - start_at

    if model is not None:
        raise NotImplementedError("model not implemented")

    if query_all is not None:
        raise NotImplementedError("query_all not implemented")

    if output_fns is not None:
        raise NotImplementedError("output_fns not implemented")

    if tar_idxs is not None:
        raise NotImplementedError("tar_idxs not implemented yet")

    step_var.assign(0)
    while True:

        inputs = {
            "values": inp_var[:,  :start_at+step_var],
        }
        idxs = inp_idxs[:, :start_at+step_var]
        # add batch indices to idxs
        batch_idxs = ein.repeat(
            tf.range(batch_size),
            'b -> b n ()',
            n=shape(inp_idxs)[1],
        )
        step_idxs = ein.repeat(
            step_var,
            ' -> b n ()',
            b=batch_size,
            n=shape(inp_idxs)[1],
        )
        scatter_idxs = tf.stack([batch_idxs, step_idxs, idxs], axis=-1)
        out_var.scatter_nd_update(
            scatter_idxs,
            
        )

        # if model is not None:
        #     outputs = model(inputs, training=False)
        #     for i, o, f in enumerate(zip(outputs, output_fns)):
        #         inp_var[i, start_at+step_var].assign(f(o))
        #         out_var[step_var, i, :start_at+step_var] = inp_var[i, :start_at+step_var]
        #         if query_all:
        #             out_var[step_var, i, start_at+step_var] = f(o)
        #             out_var[step_var, i, start_at + step_var:].assign(f(o))

        step_var.assign_add(1)
        if step_var >= n_steps:
            break

        if break_var:
            break


    return out_var

def predict_wrapper(
    name,
    desc,
    model=None,
    batch_size=None,
    seed_values=None,
    idxs=None,
    inp_output_fn=None,
    output_fns=None,
    out_var_shapes=None,
    pm=None,
    default_out_var_val=tf.constant([0.9, 0.1, 0.5]),
):
    """
    Run `predict_core` on another thread to support quick keyboard interrupt & live progress.
    Create variables for the outputs.
    """

    assert batch_size is not None or seed_values is not None, "Must provide either `batch_size` or `seed_values`"
    assert batch_size is None or seed_values is None, "Must provide either `batch_size` or `seed_values`, but not both"

    assert model is not None or seed_values is not None, "Must provide `model`, `seed_values`, or both"

    n_feature_dims = model.input_shape["context"]["values"][1]
    n_index_dims = model.input_shape["context"]["inp_idxs"][1]

    output_len = shape(idxs)[1]

    if seed_values is not None:
        assert shape(seed_values)[1] <= output_len, f"seed_values must be shorter than or the same length as than idxs, got seed_len={shape(seed_values)[1]} and idxs_len={output_len}"
        seed_len = shape(seed_values)[1]
        batch_size = shape(seed_values)[0]
    elif batch_size is not None:
        seed_values = tf.zeros([batch_size, 0, n_feature_dims], dtype=u.dtype())
        seed_len = 0

    if model is not None:
        raise NotImplementedError("model not implemented yet")

    if inp_output_fn is None:
        inp_output_fn = u.v_to_rgb_grayscale

    if output_fns is not None:
        raise NotImplementedError("output_fns not implemented yet")
        # output_fns = [u.v_to_rgb_grayscale] * batch_size

    with ExitStack() as stack:
        if pm is not None:
            stack.enter_context(pm.enter_progbar(name, desc, delete_on_success=True))

        n_steps = output_len
        inp_var = tf.Variable(
            initial_value=tf.zeros([batch_size, output_len, n_feature_dims], dtype=u.dtype()),
            trainable=False,
            name="inp_var",
        )
        if seed_values is not None:
            inp_var[:, :seed_len].assign(seed_values)

        out_var = tf.Variable(
            initial_value=ein.repeat(
                default_out_var_val,
                'c -> bat steps out c',
                bat=batch_size,
                steps=n_steps,
                out=output_len,
            ),
            trainable=False,
            name="out_var",
        )
        tar_idxs = idxs

        step_var = tf.Variable(0, dtype=tf.int32, trainable=False, name="step_var")
        break_var = tf.Variable(False, dtype=tf.bool, trainable=False, name="break_var")

        start_at = tf.constant(seed_len)
        predict_core(
            model=model,
            start_at=start_at,
            step_var=step_var,
            break_var=break_var,
            output_len=output_len,
            inp_output_fn=inp_output_fn,
            output_fns=output_fns,
            inp_var=inp_var,
            inp_idxs=idxs,
            tar_idxs=tar_idxs,
            query_all=None,
        )

        return Video(
            seed_inputs=inp_var,
            sampling_order=tar_idxs,
            mean_prediction=out_var,
            mean_entropies=None,
            sampled_predictions=None,
            sampled_entropies=None,
        )

def predict(
    output_len,
    inputs=None,
    model=None,
    seed_len=None,
    model_supports_querying: Literal[True, False, "yes_and_query_all"] = False,
    model_produces_distribution: bool = False,
    pm: Progress | Literal[True] | None = None,
):
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

    assert output_len > 0, f"output_len must be > 0, got output_len={output_len}"

    u.validate(inputs, "inputs", {
        "values": tf.TensorSpec([None, None, None], u.dtype()),
    })

    inputs = Box(inputs)

    data = inputs.data
    batch_size = shape(data)[0]
    seq_len = shape(data)[1]
    n_feature_dims = shape(data)[2]

    if model is not None:
        u.validate(inputs, "inputs", {
            "seq_idxs": tf.TensorSpec([None, None, None], tf.int32),
        })
        seq_idxs = inputs.seq_idxs
        assert shape(seq_idxs)[0] == batch_size, f"seq_idxs must have same batch size as data, got {shape(seq_idxs)[0]} ≠ {batch_size}"
        assert shape(seq_idxs)[1] == seq_len, f"seq_idxs must have same seq len as data, got {shape(seq_idxs)[1]} ≠ {seq_len}"
        n_index_dims = shape(seq_idxs)[2]

        if model_supports_querying is not False:


    outputs = Box(default_box=True)

    with ExitStack() as stack:
        if pm is True:
            pm = create_progress_manager(u.get_run_name())

        if pm is not None:
            stack.enter_context(pm.enter_spinner("Predict", "Predicting...", delete_on_success=True))

        if inputs is not None:

            outputs["Examples"]["Video"] = predict_wrapper(
                name="Examples",
                desc="Making examples animation...",
                seed_values=inputs.values,
            )




        seed_input = data[:, :seed_len, :]

    n_features = data.shape[2]

    out_var = tf.Variable(tf.zeros([batch_size, output_len, n_features]))
    predict_fn(seed_input, seq_idxs, out_var)
    if u.is_debug() and tf.reduce_any(tf.math.is_nan(out_var)).numpy():
        raise ValueError(f"NaNs in output of predict_fn of {self.name}")

    return {
        "angles": out_var,
    }


if __name__ == "__main__":

    inputs = Box({
        "context": u.list_to_dict(
            Input([None, 1], u.dtype(), name="values"),
            Input([None, 1], tf.int322, name="inp_idxs"),
        ),
        "queries": u.list_to_dict(
            Input([None, 1], u.dtype(), name="values"),
            Input([None, 1], u.dtype(), name="inp_idxs"),
            Input([None, 1], u.dtype(), name="tar_idxs"),
        ),
    })

    model = Model(
        inputs=inputs,
        outputs=-1*inputs.queries.values,
        name="test_negate",
    )

    predict(
        name="test",
        desc="Test",
        model=model,
        output_len=10,
        inputs={
            "values": tf.range(10, dtype=u.dtype())[None, :, None],
            "inp_idxs": tf.range(10, dtype=u.dtype())[None, :, None],
            "tar_idxs": tf.range(10, dtype=u.dtype())[None, :, None],
        },
        seed_len=5,
        pm=True,
    )

    outputs = model(data)

    print(outputs)
