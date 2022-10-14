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

    >>> x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> get_chunk = make_get_chunk([Einshape(sequence_dims={"a":3, "b":3})], [2, 2], chunk_mode="simple")
    >>> c = get_chunk([x])
    >>> any(tf.reduce_all(c) for c in [
    ...     tf.equal(c, tf.constant([[1, 2], [4, 5]])),
    ...     tf.equal(c, tf.constant([[2, 3], [5, 6]])),
    ...     tf.equal(c, tf.constant([[4, 5], [7, 8]])),
    ...     tf.equal(c, tf.constant([[5, 6], [8, 9]])),
    ... ])
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
