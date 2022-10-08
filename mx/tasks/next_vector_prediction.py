from dataclasses import dataclass
from typing import Any, Callable
from mx import utils
from mx.embedding import TransformerAngleVectorEmbedding
from mx.layers import input_dict
from mx.tf import *
from mx.utils import Einshape, tf_scope, DSets

from mx.pipeline import Task, Embedding

@export
@dataclass
class NextUnitVectorPrediction(Task):
    """
    Predict the next vector in a sequence, as a vector of unit vectors.
    Because the outputs are (many-dimensional) vectors, this is a regression
    task only.
    
    For a distribution prediction task, use NextTokenPrediction which
    predicts a categorical distribution over a vocabulary of vectors.
    """

    def __init__(self, chunk_size: int, n_test_val_repeats: int = 100):
        super().__init__(
            name = "Next Vector Prediction",
            identifier = "next-vector-prediction",
        )

        self.chunk_size: int = chunk_size
        "Length of chunks (sequence length)"

        self.n_test_val_repeats: int = n_test_val_repeats
        """
        Number of chunks to take out of each example to make validation and testing data.
        In training, it's infinite and the number depends on the number of training steps
        and batch size.
        """
        
    
    def configure(self, embedding: Embedding) -> Any:

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"

        n_input_dims = self.ds_cfg.n_input_dims
        sequence_length = self.chunk_size

        ### Set embedding down_cfg ###
        if isinstance(embedding, TransformerAngleVectorEmbedding):
            embedding.receive_task_config(embedding.task_config_type(
                n_input_dims,
                sequence_length,
            ))
            def fn(dsets: DSets) -> DSets:
                return dsets.map(lambda x: {
                    **x,
                    "inputs": {
                        "angles": x["inputs"]["input"],
                        "input_idxs": x["inputs"]["input_idxs"],
                    },
                })
            self._adaptor = fn
        else:
            raise NotImplementedError(f"Embedding type {type(embedding)} not implemented")

    def process(self, dsets: DSets) -> DSets:

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"
        assert self._adaptor is not None, "self.adaptor was not set by task.configure(embedding)"

        element_spec = dsets.train.element_spec
        
        assert type(element_spec) is dict, "Next-vector-prediction requires a dataset of dicts"

        for k in element_spec.keys():
            assert k in dsets.test.element_spec, f"Test set was different from train set: {k} was missing"
            assert k in dsets.val.element_spec, f"Val set was different from train set: {k} was missing"
        
        assert "data" in element_spec, "Next-vector-prediction requires a dataset with a 'data' key"
        assert len(element_spec["data"].shape) == 2, f"Data for next-vector-prediction must have only a single sequence dimension, and a single feature dimension. Got shape {element_spec['data'].shape}"

        assert "seq_idxs" in element_spec, "Next-vector-prediction requires a dataset with a 'seq_idxs' key"
        assert len(element_spec["seq_idxs"].shape) == 1, f"seq_idxs for next-vector-prediction must have shape [seq]. Got shape {element_spec['seq_idxs'].shape}"

        assert element_spec["data"].shape[0] == element_spec["seq_idxs"].shape[0], f"Data and seq_idxs must have the same sequence length. Got {element_spec['data'].shape[0]} ≠ {element_spec['seq_idxs'].shape[0]}"

        data_shape = Einshape(
            sequence_dims={"seq": element_spec["data"].shape[0]},
            feature_dims={"feat": element_spec["data"].shape[1]},
        )

        seq_idxs_shape = Einshape(
            sequence_dims={"seq": element_spec["seq_idxs"].shape[0]},
            feature_dims={},
        )

        # repeat data in order to take many random chunks from each sequence
        train, test, val = dsets.destructure()
        n_train = train.cardinality().numpy()
        dsets = DSets(
            # repeat training data infinitely. Shuffle before repeat ensures
            # uniform distribution of sequences in each batch
            train=train.shuffle(n_train).repeat(),
            # Take n_repeats random chunks from each example. don't shuffle,
            # because we want the test/val runs to be repeatable.
            test=test.repeat(self.n_test_val_repeats),
            val=val.repeat(self.n_test_val_repeats),
        )

        # dset = dset.map(inspect("repeat"))

        get_chunk = make_get_chunk(
            [
                data_shape,
                seq_idxs_shape,
            ],
            chunk_size=[self.chunk_size],
        )

        def do_chunk(x):

            data, seq_idxs = get_chunk([
                x["data"],
                x["seq_idxs"],
            ])

            return {
                **x,
                "data": data,
                "seq_idxs": seq_idxs,
            }
        
        # chunk
        dsets = dsets.map(do_chunk)

        dsets = dsets.map(lambda x: {
            "inputs": {
                "input": x["data"][:-1],
                "input_idxs": x["seq_idxs"][:-1],
                "target_idxs": x["seq_idxs"][1:],
            },
            "targets": x["data"][1:],
            "extra": {
                **x["extra"],
            },
        })

        dsets = DSets(
            train=dsets.train.enumerate(),
            test=dsets.test.enumerate(),
            val=dsets.val.enumerate(),
        )
        dsets = dsets.map(lambda i, x: { **x, "extra": x["extra"] | { "i": i } })

        return self._adaptor(dsets)

    def make_final_layer(self) -> tf.keras.layers.Layer:

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"
        assert self.model_cfg is not None, "Must call task.configure(embedding) first"

        inputs = input_dict(
            Input([None, self.model_cfg.n_output_embd], name="embd"),
        )
        dense = tf.keras.layers.Dense(self.ds_cfg.n_input_dims * 2)
        def call(inputs):
            outs = dense(inputs["embd"])
            outs = ein.rearrange(outs, "... seq (feat sincos) -> ... seq feat sincos", sincos=2)
            return { "unit_vectors": outs, **inputs }

        return Model(
            inputs=inputs,
            outputs=call(inputs),
            name="outputs",
        )

    def make_loss_fn(self) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        "Angular mean-squared-error loss."

        @tf.function
        @tf_scope
        def angular_mse_loss(targets, outputs):
            "Angular mean-squared-error loss."

            unit_vectors = outputs["unit_vectors"]

            target_sin = tf.sin(targets)
            target_cos = tf.cos(targets)
            sin = unit_vectors[..., 0]
            cos = unit_vectors[..., 1]
            return tf.reduce_mean(tf.square(target_sin - sin) + tf.square(target_cos - cos))
        
        return angular_mse_loss

    def make_predict_fn(self):
        "Predict the next vector in the sequence, starting with optional seed data."

        raise NotImplementedError("TODO")


@export
def make_get_chunk(seq_dims: list[Einshape], chunk_size: list[int], seed=None):
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

    assert len(seq_dims) > 0, "Must provide at least one sequence"
    
    assert all([ e.b_shape == seq_dims[0].b_shape for e in seq_dims ]), "All sequences must have the same batch dimensions"
    assert all([ e.s_shape == seq_dims[0].s_shape for e in seq_dims ]), "All sequences must have the same sequence dimensions"
    seq_einshape = seq_dims[0]
    assert len(chunk_size) == seq_einshape.s_rank, f"Chunk size {chunk_size} must have same rank as seq_dims {seq_einshape.s_shape}"
    assert all([s > 0 for s in chunk_size]), f"Chunk size {chunk_size} must be positive"
    assert all([seq_dim is None or chunk_dim <= seq_dim for chunk_dim, seq_dim in zip(chunk_size, seq_einshape.s_shape)]), f"All dims of chunk size ({chunk_size}) must be <= seq_dims ({seq_einshape.s_shape})"

    @tf.function
    @tf_scope
    def get_chunk(seqs):
        seqs = [ tf.ensure_shape(s, e.shape) for s, e in zip(seqs, seq_dims) ]
        seqs = [
            ein.rearrange(s, f'... {e.f_str} -> ... ({e.f_str})', **e.f)
            for s, e in zip(seqs, seq_dims)
        ]

        seq_shape = tf.shape(seqs[0])[seq_einshape.b_rank:seq_einshape.b_rank+seq_einshape.s_rank]

        max_indices = seq_shape - tf.constant(chunk_size, tf.int32)
        idxs = tf.map_fn(
            lambda max_i: tf.random.uniform([], 0, max_i, dtype=tf.int32, seed=seed),
            max_indices,
        )
        idxs = idxs[None, :] + utils.multidim_indices(chunk_size, flatten=True, elide_rank_1=False)

        # extract chunks from seqs
        seqs = [
            tf.gather_nd(s, idxs)
            for s in seqs
        ]

        new_seq_dims = [
            e.cut(chunk_size)
            for e in seq_dims
        ]

        # restore shape of sequence and feature dimensions
        seqs = [
            ein.rearrange(s, f'... ({e.s_str}) ({e.f_str}) -> ... {e.s_str} {e.f_str}', **e.s, **e.f)
            for s, e in zip(seqs, new_seq_dims)
        ]

        # ensure new shape (important for ragged tensors)
        seqs = [ tf.ensure_shape(s, e.shape) for s, e in zip(seqs, new_seq_dims) ]

        return seqs
    
    return get_chunk
