from mx.prelude import *

@export
def make_get_chunk(chunk_size: int, seed=None):
    """
    Cuts chunks of size chunk_size from the input sequence x.
    Returns a new sequence of the same rank as x, with the
    sequence dimensions being cut to chunk_size.

    Does not support batching. Use tf.map_fn to batch.

    Sequence dimensions can be ragged, in which case this
    function can be used to cut non-ragged chunks, and will
    return a non-ragged sequence of the same rank as x.

    >>> x = tf.constant([1, 2, 3, 4, 5, 6])
    >>> get_chunk = make_get_chunk(chunk_size=2)
    >>> c = get_chunk([x])
    >>> all(tf.reduce_any([
    ...     tf.equal(c, tf.constant([1, 2])),
    ...     tf.equal(c, tf.constant([2, 3])),
    ...     tf.equal(c, tf.constant([3, 4])),
    ...     tf.equal(c, tf.constant([4, 5])),
    ...     tf.equal(c, tf.constant([5, 6])),
    ... ]).numpy() for _ in range(100))
    True
    """

    assert chunk_size > 0, f"Chunk size {chunk_size} must be positive"

    @tf.function
    @u.tf_scope
    def get_chunk(seqs):

        seq_len = tf.shape(seqs[0])[0]

        max_index = seq_len - chunk_size
        i = tf.random.uniform([], 0, max_index, dtype=tf.int32, seed=seed),
        idxs = tf.range(chunk_size) + i

        # extract chunks from seqs
        seqs = [
            tf.gather(s, idxs)
            for s in seqs
        ]

        seqs = [ tf.ensure_shape(s, [chunk_size] + shape(s)[1:]) for s in seqs ]

        return seqs

    return get_chunk


@export
def make_get_chunk_batched_ragged(chunk_size: int, seed=None):
    """
    Cuts chunks of size chunk_size from the input sequence x.
    Returns a new sequence of the same rank as x, with the
    sequence dimensions being cut to chunk_size.

    Sequence dimensions can be ragged, in which case this
    function can be used to cut non-ragged chunks, and will
    return a non-ragged sequence of the same rank as x.

    This version of the function supports batching.

    >>> x = tf.RaggedTensor.from_row_lengths([1, 2, 3, 4, 5, 6], [6])
    >>> get_chunk = make_get_chunk_batched_ragged(chunk_size=2)
    >>> c = get_chunk([x])
    >>> all(tf.reduce_any([
    ...     tf.equal(c, tf.constant([[1, 2]])),
    ...     tf.equal(c, tf.constant([[2, 3]])),
    ...     tf.equal(c, tf.constant([[3, 4]])),
    ...     tf.equal(c, tf.constant([[4, 5]])),
    ...     tf.equal(c, tf.constant([[5, 6]])),
    ... ]).numpy() for _ in range(100))
    True
    >>> x = tf.RaggedTensor.from_row_lengths([1, 2, 3, 4, 5, 6], [2, 4])
    >>> get_chunk = make_get_chunk_batched_ragged(chunk_size=2)
    >>> c = get_chunk([x])
    >>> all(tf.reduce_any([
    ...     tf.equal(c, tf.constant([[1, 2], [3, 4]])),
    ...     tf.equal(c, tf.constant([[1, 2], [4, 5]])),
    ...     tf.equal(c, tf.constant([[1, 2], [5, 6]])),
    ... ]).numpy() for _ in range(100))
    True
    >>> x = tf.constant([[1, 2, 3, 4, 5, 6]])
    >>> get_chunk = make_get_chunk_batched_ragged(chunk_size=2)
    >>> c = get_chunk([x])
    >>> all(tf.reduce_any([
    ...     tf.equal(c, tf.constant([[1, 2]])),
    ...     tf.equal(c, tf.constant([[2, 3]])),
    ...     tf.equal(c, tf.constant([[3, 4]])),
    ...     tf.equal(c, tf.constant([[4, 5]])),
    ...     tf.equal(c, tf.constant([[5, 6]])),
    ... ]).numpy() for _ in range(100))
    True
    """

    assert chunk_size > 0, f"Chunk size {chunk_size} must be positive"

    @tf.function
    @u.tf_scope
    def get_chunk_batched_ragged(seqs):
        nonlocal chunk_size

        batch_size = tf.shape(seqs[0])[0]
        assert seqs[0].shape.rank >= 2, f"get_chunk_batched_ragged requires both batch and sequence dimensions, got {seqs[0].shape.rank}"

        if isinstance(seqs[0], tf.RaggedTensor):
            seq_len = seqs[0].row_lengths()
        else:
            seq_len = tf.shape(seqs[0])[1]
            seq_len = tf.broadcast_to(seq_len, [batch_size])

        seq_len = tf.cast(seq_len, tf.int32)
        chunk_size_batch = tf.constant(chunk_size, dtype=tf.int32)[None]
        chunk_size_batch = tf.broadcast_to(chunk_size_batch, [batch_size])

        # initialize a batched uniform categorical distribution
        # tf.random doesn't support batched distributions, so we
        # use tensorflow_probability.distributions (tfd) instead

        max_len = tf.reduce_max(seq_len)
        logit_table = tf.linalg.band_part(tf.ones([max_len, max_len]), -1, 0)

        max_indices = seq_len - chunk_size_batch
        max_indices_logits = tf.gather(logit_table, max_indices)

        dist = tfd.Categorical(logits=max_indices_logits)

        idxs = dist.sample(seed=seed)

        idxs = idxs[:, None] + tf.range(chunk_size)[None, :]

        # extract chunks from seqs
        seqs = [
            tf.gather(s, idxs, batch_dims=1)
            for s in seqs
        ]

        # seqs = [ tf.ensure_shape(s, [batch_size, chunk_size, ]) for s in seqs ]

        return seqs

    return get_chunk_batched_ragged


if __name__ == "__main__":
    import doctest
    doctest.testmod()
