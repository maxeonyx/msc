from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from msilib import sequence
from typing import Literal
from mx.datasets import Dataset, DSets
from mx.embedding import Embedding

from mx.utils.tf import *

@dataclass
class Task(ABC):

    name: str
    "Human-readable name"

    identifier: str
    "Unique, machine-readable identifier"

    _: KW_ONLY

    does_batching: bool = False
    "Whether the task does its own batching"

    @abstractmethod
    def adapt_for(self, embedding: Embedding, dataset: DSets):
        pass

    @abstractmethod
    def loss(self, targets, predictions):
        pass

    @abstractmethod
    def predict(self, inputs):
        pass

@dataclass
class NextVectorPrediction(Task):

    n_sequence_dims: int
    "Number of dimensions to treat as sequence dimensions. The rest are treated as feature dimensions."

    sequence_length: int
    "Length of sequence to predict"

    n_batch_dims = 0
    "Number of dimensions to treat as batch dimensions. Default 0."

    output_type: Literal["vector", "unit_vectors"] = "vector"

    _: KW_ONLY

    name: str = "Next Vector Prediction"
    identifier: str = "next-vector-prediction"

    def adapt_for(self, embedding: Embedding, dataset: DSets):
        return next_vector_prediction(dataset, self)

    def loss(self, targets, outputs):
        if self.output_type == "vector":
            return tf.reduce_mean(tf.square(targets - outputs))
        elif self.output_type == "unit_vectors":
            target_sins = tf.sin(targets)
            target_coss = tf.cos(targets)
            output_sins = outputs[..., 0]
            output_coss = outputs[..., 1]
            return tf.reduce_mean(tf.square(target_sins - output_sins) + tf.square(target_coss - output_coss))

    def predict(self, inputs):
        return inputs



def make_get_chunk(seq_dims: list[Einshape], chunk_size: list[int], chunk_mode: Literal["simple", "random"], seed=None):
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

        if chunk_mode == "simple":

            max_indices = seq_shape - tf.constant(chunk_size, tf.int32)
            idxs = tf.map_fn(
                lambda max_i: tf.random.uniform([], 0, max_i, dtype=tf.int32, seed=seed),
                max_indices,
            )
            idxs = idxs[None, :] + utils.multidim_indices(chunk_size, flatten=True, elide_rank_1=False)
        elif chunk_mode == "random":
            idxs = utils.multidim_indices(seq_shape, flatten=False)
            idxs = random_subset(idxs, chunk_size, seed=seed)
        else:
            raise ValueError(f"Unknown chunk mode {chunk_mode}.")

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

def next_vector_prediction(t_cfg: NextVectorPrediction, dsets: DSets) -> tuple[DSet, DatasetShape]:
    
    element_spec = dsets.train.element_spec
    
    assert type(element_spec) is dict, "Next-vector-prediction requires a dataset of dicts"


    for k in element_spec.keys():
        assert k in dsets.test.element_spec, f"Test set was different from train set: {k} was missing"
        assert k in dsets.val.element_spec, f"Val set was different from train set: {k} was missing"
    
    assert "data" in element_spec, "Next-vector-prediction requires a dataset with a 'data' key"
    assert len(element_spec["data"].shape) == 2, f"Data for next-vector-prediction must have only a single sequence dimension, and a single feature dimension. Got shape {element_spec['data'].shape}"

    assert "seq_idxs" in element_spec, "Next-vector-prediction requires a dataset with a 'seq_idxs' key"
    assert len(element_spec["seq_idxs"].shape) == 3, f"seq_idxs for next-vector-prediction must have shape [seq, feat, index]. Got shape {element_spec['seq_idxs'].shape}"
    assert element_spec["seq_idxs"].shape[-1] == t_cfg.n_sequence_dims, f"The final dimension of seq_idxs must have shape = n_sequence_dims dimensions. Expected {t_cfg.n_sequence_dims}, got {element_spec['seq_idxs'].shape[-1]}"


    # flatten batch dims to a single batch dim, sequence dims to a single sequence dim, and feature dims to a single feature dim
    dsets = dsets.map(lambda x: {
        **x,
        "data": tf.reshape()

    # flatten angles feature dims (h, j, d) dims to single dim
    dsets = dsets.map(lambda x: { **x, "data": ein.rearrange(x["data"], f"... {shapes["data"].f_str} -> ... ({shapes["angles"].f_str})") })
    shapes["angles"] = shapes["angles"].with_feature_dims({ "vec": shapes["angles"].f_product })



    # for Next-vector-prediction task we only need the sequence indices, and there's only
    # one sequence dimension
    dsets = dsets.map(lambda x: { **x, "idxs": x["idxs"][:, 0, 0, 0, :1] })
    shapes["idxs"] = shapes["idxs"].with_feature_dims({ "i": 1 })

    # repeat data to take many random chunks from each sequence
    train, test, val = dsets.destructure()
    n_train = train.cardinality().numpy()
    dsets = DSet(
        # repeat training data infinitely
        train=train.repeat().shuffle(n_train),

        # take 10 random chunks from each example
        test=test.repeat(100),
        val=val.repeat(100),
    )

    # dset = dset.map(inspect("repeat"))

    get_chunk = make_get_chunk(
        [
            shapes["angles"],
            shapes["orig_angles"],
            shapes["idxs"],
        ],
        chunk_size=[t_cfg.sequence_length],
        chunk_mode="simple",
    )

    def do_chunk(x):

        angles, orig_angles, idxs = get_chunk([
            x["angles"],
            x["orig_angles"],
            x["idxs"],
        ])

        return {
            **x,
            "angles": angles,
            "orig_angles": orig_angles,
            "idxs": idxs,
        }
    
    # chunk
    dsets = dsets.map(do_chunk)

    # dset = dset.map(inspect("chunk"))

    dsets = dsets.map(lambda x: {
        "inputs": {
            "input": tf.identity(x["angles"][:-1], name="inputs_angles"),
            "input_idxs": tf.identity(x["idxs"][:-1], name="inputs_input_idxs"),
            "target_idxs": tf.identity(x["idxs"], name="inputs_target_idxs"),
        },
        "targets": tf.identity(x["angles"], name="targets_targets"),
        "extra": {
            "orig_angles": tf.identity(x["orig_angles"], name="extra_orig_angles"),
            "filename": tf.identity(x["filename"], name="extra_filename"),
        },
    })

    dsets = DSet(
        train=dsets.train.enumerate(),
        test=dsets.test.enumerate(),
        val=dsets.val.enumerate(),
    )
    dsets = dsets.map(lambda i, x: { **x, "extra": x["extra"] | { "i": i } })

    # set shapes to chunk size and sequence length
    shapes = DatasetShape(
        inputs={
            "input": shapes["angles"].with_sequence_dims({ "f": t_cfg.sequence_length - 1 }),
            "input_idxs": shapes["idxs"].with_sequence_dims({ "f": t_cfg.sequence_length - 1 }),
            "target_idxs": shapes["idxs"].with_sequence_dims({ "f": t_cfg.sequence_length }),
        },
        targets=shapes["angles"].with_sequence_dims({ "f": t_cfg.sequence_length }),
        extra={
            "orig_angles": shapes["orig_angles"],
            "filename": shapes["filename"],
        },
    )

    return dsets, shapes
