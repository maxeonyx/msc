from mx.prelude import *
from ._layer_utils import MxLayer
from mx.utils import Einshape


# the above as a functional layer/model
@export
def learned_mix_add(n_embd, name="mix"):
    """
    Learns a scale paramter `mix` which adds `res` and `x` such
    that the norm (/ variance) of the result is constant no matter
    the mixing factor.
    """

    mix = tf.Variable(
        name="mix",
        shape=(),
        initial_value=tf.constant(4., u.dtype()),
        trainable=True,
    )

    def call(inputs):
        nonlocal mix

        res, x = inputs
        mix = tf.math.sigmoid(mix)
        a = tf.sqrt(mix)
        b = tf.sqrt(1. - mix)

        return res * a + x * b

    inputs = [
        Input(shape=[None, n_embd], name="residual"),
        Input(shape=[None, n_embd], name="input"),
    ]

    return Model(inputs=inputs, outputs=call(inputs), name=name)

# the above as latex
# \begin{align}
#     \text{sigmoid}(m) &= \frac{1}{1 + e^{-m}} \\
#     \text{lma}(a, b, m) &= \sqrt{\text{sigmoid}(m)} a + \sqrt{1 - \text{sigmoid}(m)} b \\
#     \text{residual}(x, f, m) &= \text{lma}(x, f(x), m)
# \end{align}

# with "Noise Gating"




def residual(embd_shape: Einshape, n_layers: int=None, make_layer: Callable[[int], MxLayer]=None, layers: list[MxLayer]=None, normalization='scale', dropout=0.1, name="residual") -> Model:

    making_layers = n_layers is not None and make_layer is not None
    using_layers = layers is not None

    assert not (making_layers and using_layers), "Cannot specify both `layers` and `n_layers` + `make_layer`"
    assert making_layers or using_layers, "Must specify either `layers` or `n_layers` + `make_layer`"

    if making_layers:
        layers = [make_layer(i) for i in range(n_layers)]

    def call(embd):
        for layer in layers:
            embd += layer(embd)
        return embd

    inputs = u.input_dict(
        Input(shape=embd_shape.s_f_shape, name="embd")
    )

    return Model(inputs=inputs, outputs=call(inputs), name=name)
