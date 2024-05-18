from mx.prelude import *

@export
class LearnedMixAdd(u.MxModule):
    """
    Learns a scale paramter `mix` which adds `res` and `x` such
    that the norm (/ variance) of the result is constant no matter
    the mixing factor.

    >>> x = tf.constant([0., 1.], dtype=u.dtype())
    >>> res = tf.constant([1., 0.], dtype=u.dtype())
    >>> print(LearnedMixAdd(start_value=0.)([res, x]).numpy())
    [0.70710677 0.70710677]
    >>> print(LearnedMixAdd(start_value=1e10)([res, x]).numpy())
    [1. 0.]
    >>> print(LearnedMixAdd()([res, x]).numpy())
    [0.9909661  0.13411278]
    """

    @u.tf_scope
    def __init__(
        self,
        start_value: float = 4.,
        name="mix",
    ) -> None:
        super().__init__(
            name=name,
            desc=f"LearnedMixAdd{f'({start_value})' if start_value != 4. else ''}",
        )
        self.start_value = start_value

        self.mix = tf.Variable(
            name="mix",
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(self.start_value),
            trainable=True,
        )

    @tf.function
    @u.tf_scope
    def __call__(self, inputs):

        residual, x = inputs

        mix = tf.cast(tf.math.sigmoid(self.mix), u.dtype())
        a = tf.sqrt(mix)
        b = tf.sqrt(1. - mix)

        return residual * a + x * b


if __name__ == "__main__":
    import doctest
    doctest.testmod()
