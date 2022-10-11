from mx.prelude import *

from ._blocks import featurewise_dense
from mx import layers
from mx.utils import Einshape

@dataclass
class PredictionHead(tf.Module):
    final_layer: tf.Module | None
    """
    The final learned layer, eg. logits for classification, or parameters for a distribution.
    For regression, this is None unless there is a special loss function.
    """

    loss_fn: tf.Module
    """
    The loss function. Takes two parameters `targets` and:
    - regression: `output`
    - classification: `logits`
    - distribution: `params`
    """

    output_fns: dict[str, tf.Module] | None
    """
    Functions that transform the model output eg. sampling or distribution statistics.
    Typically None for regression, but can be used for classification or distribution outputs.
    """


def mse(in_dims: Einshape, name="mse") -> PredictionHead:

    out_dims = in_dims


    loss_fn_inputs = u.input_dict(
        Input(shape=out_dims.s_f_shape, name="targets"),
        Input(shape=out_dims.s_f_shape, name="output"),
    )
    def loss_fn_call(targets, output):
        return tf.reduce_mean(tf.square(targets - output))
    loss_fn = Model(
        inputs=loss_fn_inputs,
        outputs=loss_fn_call(loss_fn_inputs),
        name="loss",
    )

    return PredictionHead(
        final_layer=None,
        loss_fn=loss_fn,
        output_fns=None,
        name=name,
    )


def circular_mse(target_dims: Einshape, embd_dims: Einshape, name="mse") -> PredictionHead:
    """
    Circular mean squared error. Takes targets as angles in radians, and y_pred as unit vectors.
    """

    out_dims = target_dims.append_feature_dim("sincos", 2)

    final_layer_inputs = u.input_dict(
        Input(shape=embd_dims.s_f_shape, name="embd"),
    )
    to_sincos_dense = featurewise_dense(
        in_dims=embd_dims,
        out_dims=out_dims,
        name="to_sincos",
    )
    def final_layer_call(embd, **kwargs):
        return { "unit_vectors": to_sincos_dense(embd), **kwargs }
    final_layer = Model(
        inputs=final_layer_inputs,
        outputs=final_layer_call(final_layer_inputs),
        name="unit_vectors",
    )

    loss_fn_inputs = u.input_dict(
        Input(shape=target_dims.s_f_shape, name="targets"),
        Input(shape=out_dims.s_f_shape, name="unit_vectors"),
    )
    def loss_fn_call(inputs):
        targets = inputs["targets"]
        unit_vectors = inputs["unit_vectors"]

        x = unit_vectors[..., 0]
        y = unit_vectors[..., 1]
        return tf.reduce_mean(tf.square(tf.math.sin(targets) - y) + tf.square(tf.math.cos(targets) - x))

    loss_fn = Model(
        inputs=loss_fn_inputs,
        outputs=loss_fn_call(loss_fn_inputs),
        name="loss",
    )

    to_angles_inputs = u.input_dict(
        Input(shape=out_dims.s_f_shape, name="unit_vectors"),
    )
    def to_angles_call(inputs):
        unit_vectors = inputs["unit_vectors"]
        x = unit_vectors[..., 0]
        y = unit_vectors[..., 1]
        return tf.math.atan2(y, x)
    output_fns = {
        "Angle": Model(
            inputs=to_angles_inputs,
            outputs=to_angles_call(to_angles_inputs),
            name="angles",
        ),
    }

    return PredictionHead(
        final_layer=final_layer,
        loss_fn=loss_fn,
        output_fns=output_fns,
    )

def categorical(in_dims: Einshape, num_categories: int, name="categorical") -> PredictionHead:

    out_dims = in_dims.with_feature_dims({ "c": num_categories })

    final_layer_inputs = u.input_dict(
        Input(shape=in_dims.s_f_shape, name="embd"),
    )
    to_logits = featurewise_dense(
        in_dims=in_dims,
        out_dims=out_dims,
        name="logits",
    )
    def final_layer_call(embd, **kwargs):
        return { "logits": to_logits(embd), **kwargs }
    final_layer = Model(
        inputs=final_layer_inputs,
        outputs=final_layer_call(final_layer_inputs),
        name="logits",
    )

    def dist(logits):
        return tfd.Categorical(logits=logits)

    loss_fn_inputs = u.input_dict(
        Input(shape=out_dims.s_f_shape, name="targets"),
        Input(shape=out_dims.s_f_shape, name="logits"),
    )
    def loss_fn_call(targets, logits):
        d = dist(logits)
        return -tf.reduce_mean(d.log_prob(targets))
    loss_fn = Model(
        inputs=loss_fn_inputs,
        outputs=loss_fn_call(loss_fn_inputs),
        name="loss",
    )

    def mode_call(logits):
        return tf.argmax(logits, axis=-1)

    def sample_call(logits):
        return dist(logits).sample()

    def entropy_call(logits):
        return dist(logits).entropy()

    logits = u.input_dict(
        Input(shape=out_dims.s_f_shape, name="logits"),
    )

    output_fns = {
        "Mode": Model(
            inputs=logits,
            outputs=mode_call(logits),
            name="mode",
        ),
        "Sample": Model(
            inputs=logits,
            outputs=sample_call(logits),
            name="sample",
        ),
        "Entropy": Model(
            inputs=logits,
            outputs=entropy_call(logits),
            name="entropy",
        ),
    }

    return PredictionHead(
        final_layer=final_layer,
        loss_fn=loss_fn,
        output_fns=output_fns,
        name=name,
    )
