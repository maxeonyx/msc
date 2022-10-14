from mx.prelude import *
from mx.utils import Einshape

@export
def random_subset(options, n, seed=None):
    options = tf.random.shuffle(options, seed=seed)
    return options[:n]

@export
@u.tf_scope
def weighted_random_n(n_max, D=2., B=10., seed=None):
    """
    Picks random indices from [0, n_max) with skewed distribution
    towards the lower numbers.
    p(index i) = 1/D * p(index Di) = 1/D**2 * p(index D**2i)
    The probability of each index decreases by D every power of b
    """

    B, D, n_max = tf.cast(B, tf.float32), tf.cast(
        D, tf.float32), tf.cast(n_max, tf.float32)

    def logb1p(b, x): return tf.math.log1p(x)/tf.math.log(b)
    # probabilities
    # probabilities = l1_norm(powf(D, -logb1p(B, tf.range(n_max))))
    # log probabilities base e
    log_probabilities = -logb1p(B, tf.range(n_max))*tf.math.log(D)

    return tf.random.categorical(log_probabilities[None, :], 1, seed=seed)[0, 0]

@export
def make_decimate(threshold: float, dims: Einshape, other_params: list[tuple[Einshape, ...]] = []):
    """
    Cut down a sequence along a single `seq_dim` to the regions where the feature
    values vary, as measured by the L2 norm across `feat_dims`.

    Does not support batching, use tf.map_fn to apply to a batch.
    """

    other_dims = [d for d, t in other_params]

    if len(other_dims) == 0:
        input_signature=[
            tf.TensorSpec(shape=dims.s_f_shape, dtype=tf.float32),
        ]
    else:
        other_seqs_tspec = tuple([tf.TensorSpec(shape=d.s_f_shape, dtype=t) for d, t in other_params])
        input_signature = [
            tf.TensorSpec(shape=dims.s_f_shape, dtype=tf.float32),
            *other_seqs_tspec,
        ]

    @tf.function(input_signature=input_signature)
    @u.tf_scope
    def decimate(data, *other_data):
        data = ein.rearrange(data, f'f {dims.f_str} -> f ({dims.f_str})', **dims.f_dict)
        other_data = [ein.rearrange(o_data, f'f {o.f_str} -> f ({o.f_str})', **o.f_dict) for o_data, o in zip(other_data, other_dims)]
        len_data = tf.shape(data)[0]
        decimated_data = data[:1, :]
        decimated_other_data = [o_data[:1, :] for o_data in other_data]

        def decimate_step(i, decimated_data, decimated_other_data):
            decimated_data, decimated_other_data = tf.cond(
                pred=tf.greater(tf.linalg.norm(data[i] - decimated_data[-1]), threshold),
                true_fn=lambda: (
                    tf.concat([decimated_data, data[i:i+1]], axis=0),
                    [tf.concat([dec_o_data, o_data[i:i+1]], axis=0) for dec_o_data, o_data in zip(decimated_other_data, other_data)]
                ),
                false_fn=lambda: (decimated_data, decimated_other_data),
            )
            return i+1, decimated_data, decimated_other_data

        _i, decimated_data, decimated_other_data = tf.while_loop(
            lambda i, decimated_data, decimated_other_data: tf.less(i, len_data),
            decimate_step,
            [tf.constant(1), decimated_data, decimated_other_data],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, *shape(data)[1:]]),
                [tf.TensorShape([None, *shape(o_data)[1:]]) for o_data in other_data],
            ],
        )

        decimated_data = ein.rearrange(decimated_data, f'f ({dims.f_str}) -> f {dims.f_str}', **dims.f_dict)
        decimated_other_data = [ein.rearrange(o_data, f'f ({o.f_str}) -> f {o.f_str}', **o.f_dict) for o_data, o in zip(decimated_other_data, other_dims)]
        return decimated_data, decimated_other_data

    if len(other_params) == 0:
        def decimate(data):
            d, _other_d = decimate(data)
            return d

    return decimate
