from functools import reduce
from mx.prelude import *

_TEST_REPS = 3

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

    >>> get_chunk = make_get_chunk(chunk_size=2)
    >>> x = tf.constant([1, 2, 3, 4, 5, 6])
    >>> all(
    ...     do_f_and_assert_any(get_chunk, [x], vals=tf.constant([
    ...         [1, 2],
    ...         [2, 3],
    ...         [3, 4],
    ...         [4, 5],
    ...         [5, 6],
    ...     ]))
    ...     for _ in range(_TEST_REPS)
    ... )
    True
    >>> x = tf.constant([[1], [2], [3], [4], [5], [6]])
    >>> all(
    ...     do_f_and_assert_any(get_chunk, [x], vals=tf.constant([
    ...         [[1], [2]],
    ...         [[2], [3]],
    ...         [[3], [4]],
    ...         [[4], [5]],
    ...         [[5], [6]],
    ...     ]))
    ...     for _ in range(_TEST_REPS)
    ... )
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

        ic(shape(seqs[0]))

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

    >>> get_chunk = make_get_chunk_batched_ragged(chunk_size=2)
    >>> x = tf.RaggedTensor.from_row_lengths([1, 2, 3, 4, 5, 6], [6])
    >>> all(
    ...     do_f_and_assert_any(get_chunk, [x], vals=tf.constant([
    ...         [[1, 2]],
    ...         [[2, 3]],
    ...         [[3, 4]],
    ...         [[4, 5]],
    ...         [[5, 6]],
    ...     ]))
    ...     for _ in range(_TEST_REPS)
    ... )
    True
    >>> x = tf.RaggedTensor.from_row_lengths([1, 2, 3, 4, 5, 6], [2, 4])
    >>> all(
    ...     do_f_and_assert_any(get_chunk, [x], vals=tf.constant([
    ...         [[1, 2], [3, 4]],
    ...         [[1, 2], [4, 5]],
    ...         [[1, 2], [5, 6]],
    ...     ]))
    ...     for _ in range(_TEST_REPS)
    ... )
    True
    >>> x = tf.constant([[1, 2], [3, 4], [5, 6]])
    >>> all(
    ...     do_f_and_assert_any(get_chunk, [x], vals=tf.constant([
    ...         [[1, 2], [3, 4], [5, 6]],
    ...     ]))
    ...     for _ in range(_TEST_REPS)
    ... )
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
        chunk_size_batch = tf.broadcast_to(chunk_size, [batch_size])
        tf.print(seq_len),
        tf.print(chunk_size_batch)
        # initialize a batched uniform categorical distribution
        # tf.random doesn't support batched distributions, so we
        # use tensorflow_probability.distributions (tfd) instead

        max_len = tf.reduce_max(seq_len)
        logit_table = tf.linalg.band_part(tf.ones([max_len, max_len]), -1, 0)
        tf.print(logit_table)

        probability_table = logit_table / tf.reduce_sum(logit_table, axis=1, keepdims=True)

        max_indices = seq_len - chunk_size_batch
        max_indices_probs = tf.gather(probability_table, max_indices)
        tf.print(max_indices_probs)
        dist = tfd.Categorical(probs=max_indices_probs)

        idxs = dist.sample(seed=seed)
        tf.print(idxs)
        idxs = idxs[:, None] + tf.range(chunk_size)[None, :]

        # extract chunks from seqs
        seqs = [
            tf.gather(s, idxs, axis=1, batch_dims=1)
            for s in seqs
        ]

        # seqs = [ tf.ensure_shape(s, [batch_size, chunk_size, ]) for s in seqs ]

        return seqs

    return get_chunk_batched_ragged


@export
def make_get_random_slices_batched_ragged(n_slices, seed=None):
    """
    Given some parallel input sequences, select `n_slices` random indices in random order
    from along their shared sequence dimension, then gather the results back into dense
    parallel tensors.

    >>> get_slices = make_get_random_slices_batched_ragged(n_slices=2)
    >>> x = tf.constant([[1, 2, 3]])
    >>> all(
    ...     do_f_and_assert_any(get_slices, [x], vals=tf.constant([
    ...         [[1, 2]],
    ...         [[1, 3]],
    ...         [[2, 1]],
    ...         [[2, 3]],
    ...         [[3, 1]],
    ...         [[3, 2]],
    ...     ]))
    ...     for _ in range(_TEST_REPS)
    ... )
    True
    >>> all(
    ...     not do_f_and_assert_any(get_slices, [x], vals=tf.constant([
    ...         [[1, 1]],
    ...         [[2, 2]],
    ...         [[3, 3]],
    ...     ]))
    ...     for _ in range(_TEST_REPS)
    ... )
    True
    """

    @tf.function
    @u.tf_scope
    def get_random_slices_batched_ragged(seqs):
        nonlocal n_slices

        batch_size = tf.shape(seqs[0])[0]
        assert seqs[0].shape.rank >= 2, f"get_random_slices_batched_ragged requires both batch and sequence dimensions, got {seqs[0].shape.rank}"

        if isinstance(seqs[0], tf.RaggedTensor):
            seq_lens = seqs[0].row_lengths()
        else:
            seq_lens = tf.shape(seqs[0])[1]
            seq_lens = tf.broadcast_to(seq_lens, [batch_size])

        seq_lens = tf.cast(seq_lens, tf.int32)

        def get_random_slice(seq_len, n):
            """
            Create a range of indices, shuffle them then take n
            """
            idxs = tf.range(seq_len)
            idxs = tf.random.shuffle(idxs, seed=seed)
            return idxs[:n]

        idxs = tf.map_fn(
            lambda x: get_random_slice(x, n_slices),
            seq_lens,
        )

        # gather slices
        seqs = [
            tf.gather(s, idxs, batch_dims=1)
            for s in seqs
        ]

        return seqs

    return get_random_slices_batched_ragged




def do_f_and_assert_any(f, *args, vals):
    actual = f(*args)
    reduce_all_axes = tf.range(tf.rank(vals))[1:]
    # reduce_any_axes = the rest
    return tf.reduce_any(tf.reduce_all(tf.equal(*ic((actual, vals))), axis=reduce_all_axes))



if __name__ == "__main__":
    import doctest
    doctest.testmod()
