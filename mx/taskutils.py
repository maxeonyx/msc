from queue import Empty
from mx.prelude import *


@export
class RandomChunk(u.MxLayer):
    """
    Cuts chunks of size chunk_size from the input sequence x.
    Returns a new sequence of the same rank as x, with the
    sequence dimensions being cut to chunk_size.

    Sequence dimensions can be ragged, in which case this
    function can be used to cut non-ragged chunks, and will
    return a non-ragged sequence of the same rank as x.

    This version of the function supports batching.

    >>> get_chunk = RandomChunk(chunk_size=2)
    >>> x = { 'v': tf.RaggedTensor.from_row_lengths([1, 2, 3, 4, 5, 6], [6]) }
    >>> all(
    ...     do_f_and_assert_any(get_chunk, x, v=tf.constant([
    ...         [[1, 2]],
    ...         [[2, 3]],
    ...         [[3, 4]],
    ...         [[4, 5]],
    ...         [[5, 6]],
    ...     ]))
    ...     for _ in range(100)
    ... )
    True
    >>> x = { 'v': tf.RaggedTensor.from_row_lengths([1, 2, 3, 4, 5, 6], [2, 4]) }
    >>> all(
    ...     do_f_and_assert_any(get_chunk, x, v=tf.constant([
    ...         [[1, 2], [3, 4]],
    ...         [[1, 2], [4, 5]],
    ...         [[1, 2], [5, 6]],
    ...     ]))
    ...     for _ in range(100)
    ... )
    True
    >>> x = { 'v': tf.constant([[1, 2], [3, 4], [5, 6]]) }
    >>> all(
    ...     do_f_and_assert_any(get_chunk, x, v=tf.constant([
    ...         [[1, 2], [3, 4], [5, 6]],
    ...     ]))
    ...     for _ in range(100)
    ... )
    True
    """

    def __init__(
        self,
        chunk_size: int,
        seed=None,
        name="chunk",
        desc=None,
    ):
        if desc is None:
            desc = f"Random chunks of size {chunk_size}"
        super().__init__(name=name, desc=desc)

        assert chunk_size > 0, f"Chunk size {chunk_size} must be positive"

        self.chunk_size = chunk_size
        self.seed = seed

    @u.tf_scope
    @tf.function
    def call(self, seqs):
        seq1 = list(seqs.values())[0]
        batch_size = tf.shape(seq1)[0]
        assert seq1.shape.rank >= 2, f"get_chunk_batched_ragged requires both batch and sequence dimensions, got {seqs[0].shape.rank}"

        if isinstance(seq1, tf.RaggedTensor):
            seq_len = seq1.row_lengths()
        else:
            seq_len = tf.shape(seq1)[1]
            seq_len = tf.broadcast_to(seq_len, [batch_size])

        seq_len = tf.cast(seq_len, tf.int32)
        chunk_size_batch = tf.broadcast_to(self.chunk_size, [batch_size])
        # dbg(seq_len, "get_chunk_batched_ragged - seq_len")
        # dbg(chunk_size_batch, "get_chunk_batched_ragged - chunk_size_batch")
        # initialize a batched uniform categorical distribution
        # tf.random doesn't support batched distributions, so we
        # use tensorflow_probability.distributions (tfd) instead

        max_len = tf.reduce_max(seq_len)
        logit_table = tf.linalg.band_part(tf.ones([max_len, max_len]), -1, 0)
        # dbg(logit_table, "get_chunk_batched_ragged - logit_table")

        probability_table = logit_table / tf.reduce_sum(logit_table, axis=1, keepdims=True)

        max_indices = seq_len - chunk_size_batch
        max_indices_probs = tf.gather(probability_table, max_indices)
        # dbg(max_indices_probs, "get_chunk_batched_ragged - max_indices_probs")
        dist = tfd.Categorical(probs=max_indices_probs)

        idxs = dist.sample(seed=self.seed)
        # dbg(idxs, "get_chunk_batched_ragged - idxs")
        idxs = idxs[:, None] + tf.range(self.chunk_size)[None, :]

        # extract chunks from seqs
        seqs = {
            name: tf.gather(s, idxs, axis=1, batch_dims=1)
            for name, s in seqs.items()
        }

        return seqs


@export
class RandomSlices(u.MxLayer):
    """
    Given some parallel input sequences, select `n_slices` random indices in random order
    from along their shared sequence dimension, then gather the results back into dense
    parallel tensors.

    >>> get_slices = RandomSlices(n_slices=2)
    >>> x = { 'v': tf.constant([[1, 2, 3]]) }
    >>> all(
    ...     do_f_and_assert_any(get_slices, x, v=tf.constant([
    ...         [[1, 2]],
    ...         [[1, 3]],
    ...         [[2, 1]],
    ...         [[2, 3]],
    ...         [[3, 1]],
    ...         [[3, 2]],
    ...     ]))
    ...     for _ in range(100)
    ... )
    True
    >>> all(
    ...     not do_f_and_assert_any(get_slices, x, v=tf.constant([
    ...         [[1, 1]],
    ...         [[2, 2]],
    ...         [[3, 3]],
    ...     ]))
    ...     for _ in range(100)
    ... )
    True
    """

    def __init__(
        self,
        n_slices: int,
        seed=None,
        name="slices",
        desc=None,
    ):
        if desc is None:
            desc = f"Random slices of size {n_slices}"
        super().__init__(name=name, desc=desc)

        assert n_slices > 0, f"n_slices {n_slices} must be positive"

        self.n_slices = n_slices
        self.seed = seed

    @tf.function
    @u.tf_scope
    def call(self, seqs):
        seq1 = list(seqs.values())[0]
        batch_size = tf.shape(seq1)[0]
        assert seq1.shape.rank >= 2, f"get_random_slices_batched_ragged requires both batch and sequence dimensions, got {seqs[0].shape.rank}"

        if isinstance(seq1, tf.RaggedTensor):
            seq_lens = seq1.row_lengths()
        else:
            seq_lens = tf.shape(seq1)[1]
            seq_lens = tf.broadcast_to(seq_lens, [batch_size])

        seq_lens = tf.cast(seq_lens, tf.int32)

        def get_random_slice(seq_len, n):
            """
            Create a range of indices, shuffle them then take n
            """
            idxs = tf.range(seq_len)
            idxs = tf.random.shuffle(idxs, seed=self.seed)
            return idxs[:n]

        idxs = tf.map_fn(
            lambda x: get_random_slice(x, self.n_slices),
            seq_lens,
        )

        # gather slices
        seqs = {
            name: tf.gather(s, idxs, batch_dims=1)
            for name, s in seqs.items()
        }

        return seqs


@export
@u.tf_scope
def discretize(values, codebook, dist_fn=u.dist):
    """
    Discretize the given values using the given codebook.

    values and codebook should have the same shape, except that the
    first dimensions can be different lengths.

    >>> codebook = tf.constant([[0], [1], [5]])
    >>> values = tf.constant([[0.3], [1.6], [2.], [3.5], [4.5], [5.5]])
    >>> discretize(values, codebook).numpy()
    array([0, 1, 1, 2, 2, 2])
    """
    codebook = tf.cast(tf.convert_to_tensor(codebook), u.dtype())
    values = tf.cast(tf.convert_to_tensor(values), u.dtype())

    feat_shape = codebook.shape[1:]
    feat_dim_names = [f"feat_{i}" for i in range(len(feat_shape))]
    feat_dim_ein = " ".join(feat_dim_names)  # "feat_0 feat_1 feat_2" etc.
    feat_dims = {
        name: s
        for name, s in zip(feat_dim_names, feat_shape)
    }
    batch_shape  = values.shape[:-len(feat_shape)]
    batch_dim_names = [f"batch_{i}" for i in range(len(batch_shape))]
    batch_dim_ein = " ".join(batch_dim_names)  # "batch_0 batch_1" etc.
    empty_batch_dims = {
        name: 1
        for name in batch_dim_names
    }
    values = ein.rearrange(
        values,
        f'... -> () ...',
    )
    codebook = ein.repeat(
        codebook,
        f'cblen {feat_dim_ein} -> cblen {batch_dim_ein} {feat_dim_ein}',
        **empty_batch_dims,
        **feat_dims,
    )
    dists = dist_fn(values, codebook)
    tokens = tf.argmin(dists, axis=0)
    return tokens


@export
class Discretize(u.MxLayer):
    """
    Vectors to token with argmin and a codebook.

    >>> codebook = tf.constant([[0], [1], [5]])
    >>> values = tf.constant([[0.3], [1.6], [2.], [3.5], [4.5], [5.5]])
    >>> Discretize(codebook)({
    ...     'values': values,
    ... })
    {'values': <tf.Tensor: shape=(6,), dtype=int64, numpy=array([0, 1, 1, 2, 2, 2])>}
    >>> values = tf.constant([[[0.3], [1.6], [2.], [3.5], [4.5], [5.5]]])
    >>> Discretize(codebook)({
    ...     'values': values,
    ... })
    {'values': <tf.Tensor: shape=(1, 6), dtype=int64, numpy=array([[0, 1, 1, 2, 2, 2]])>}
    >>> codebook = tf.constant([[0., 0.], [1., 0.], [1., 1.], [0., 1]])
    >>> values = tf.constant([[[0.3, 0.3], [1.2, 0.2], [0.2, 1.2], [-.5, -.5], [1.1, 1.1], [5.5, 5.5]]])
    >>> Discretize(codebook)({
    ...     'values': values,
    ... })
    {'values': <tf.Tensor: shape=(1, 6), dtype=int64, numpy=array([[0, 1, 3, 0, 2, 2]])>}
    """

    def __init__(
        self,
        codebook,
        dist_fn=u.dist,
        name='discretize',
        desc=None,
        **kwargs,
    ):
        if desc is None:
            desc = f'Discretize into {len(codebook)} tokens'
        super().__init__(
            name=name,
            desc=desc,
            **kwargs,
        )
        self.codebook = codebook
        self.dist_fn = dist_fn

    def compute_output_signature(self, input_signature):
        batch_size = input_signature['values'].shape[0]
        seq_shape = input_signature['values'].shape[1:-1]
        # feat dim is removed
        return {
            **input_signature,
            'values': tf.TensorSpec([batch_size, *seq_shape], tf.int32),
        }

    @u.tf_scope
    def call(self, inputs):
        return {
            **inputs,
            'values': discretize(inputs['values'], self.codebook, self.dist_fn),
        }


def do_f_and_assert_any(f, *args, v):
    actual = f(*args)['v']
    reduce_all_axes = tf.range(tf.rank(v))[1:]
    # reduce_any_axes = the rest
    return tf.reduce_any(tf.reduce_all(tf.equal(actual, v), axis=reduce_all_axes))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
