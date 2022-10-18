from mx.prelude import *
from ._layer_utils import MxLayer
from mx.utils import Einshape

@export
class LearnedMixAdd(layers.Layer):
    """
    A custom add operator for a residual block, which learns a mixing coefficient,
    and scales the residual to have the same variance as the input.

    """

    def __init__(self, name="mix"):
        super().__init__(name=name)
    def build(self, input_shape):
        self.mix = self.add_weight(
            name="mix",
            shape=(),
            initializer=keras.initializers.Constant(4.),
            trainable=True,
        )
        """The amount of residual to mix in."""

    def call(self, inputs):
        res, x = inputs
        mix = tf.math.sigmoid(self.mix)
        a = tf.sqrt(mix)
        b = tf.sqrt(1. - mix)

        return res * a + x * b


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
