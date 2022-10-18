from mx.embeddings.vector_embeddings import DebugCodebook, Multidim_TaskConfig, VectorCodebookMultidim
from mx.prelude import *

from mx.utils import DSets, dtype, inspect, is_debug
from mx.tasks.utils import make_get_chunk, make_get_chunk_batched_ragged
from mx.embeddings import AngleVectorSequence, VectorSinusoidalMultidim
from mx.pipeline import Task, MxEmbedding, Task_DatasetConfig


@export
@dataclass
class VectorSequenceMSE_DatasetConfig(Task_DatasetConfig):
    f"""
    Dataset-specific configuration for VectorSequenceMSE
    """
    seq_dims: list[int]

@export
@dataclass
class VectorSequenceMSE(Task):
    """
    Takes a sequence of vectors and their N-dimensional indices,
    cuts them into chunks of length `chunk_size`, and yields an infinite
    stream of chunks.

    Mean-squared-error loss.
    """


    def __init__(self, chunk_size: int, pred_seed_len: int = 0, pred_output_len: int = None, n_test_val_repeats: int = 100):
        super().__init__(
            name = "Vector Sequence MSE",
            identifier = "vector-sequence-mse",
        )

        self.chunk_size: int = chunk_size
        "Length of chunks (sequence length)"

        self.n_test_val_repeats: int = n_test_val_repeats
        """
        Number of chunks to take out of each example to make validation and testing data.
        In training, it's infinite and the number depends on the number of training steps
        and batch size.
        """

        self.pred_seed_len: int = pred_seed_len
        """
        Default amount of seed data to use when predicting output sequences.
        """

        self.pred_output_len: int = chunk_size if pred_output_len is None else pred_output_len
        """
        Default length of predicted sequences.
        """

        ## set by dataset.configure(task) ##
        self.ds_config_type: Type[VectorSequenceMSE_DatasetConfig] = VectorSequenceMSE_DatasetConfig
        "Required dataset-specific config"

        self.ds_cfg: VectorSequenceMSE_DatasetConfig = None

        # self.model_config_type: Type[Task.ModelSpecificConfig] = Task.ModelSpecificConfig
        # "Required model-specific config"

        assert self.pred_seed_len < self.pred_output_len, f"pred_seed_len must be less than pred_output_len. Got pred_seed_len={self.pred_seed_len} and pred_output_len={self.pred_output_len}"


    def configure(self, embedding: MxEmbedding):

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"

        ### Set embedding down_cfg ###
        if isinstance(embedding, VectorCodebookMultidim):
            embedding.receive_task_config(embedding.task_config_type(
                n_input_dims=self.ds_cfg.n_input_dims,
                seq_len=self.chunk_size,
                seq_dims=self.ds_cfg.seq_dims,
            ))
            def adapt_in(x):
                return {
                    **x,
                    "inputs": {
                        "values": x["inputs"]["values"],
                        "seq_idxs": x["inputs"]["seq_idxs"],
                    }
                }
            self.adapt_in = adapt_in
        elif embedding.task_config_type == Multidim_TaskConfig:
            embedding.receive_task_config(embedding.task_config_type(
                n_input_dims=self.ds_cfg.n_input_dims,
                seq_dims=self.ds_cfg.seq_dims,
            ))
            def adapt_in(x):
                return {
                    **x,
                    "inputs": {
                        "values": x["inputs"]["values"],
                        "seq_idxs": x["inputs"]["seq_idxs"],
                    }
                }
            self.adapt_in = adapt_in
        else:
            raise NotImplementedError(f"Task {type_name(self)} does not support Embedding {type_name(embedding)}. If using autoreload in IPython, try restarting the interpreter.")

    def process(self, dsets: DSets) -> DSets:

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"
        assert self.adapt_in is not None, "Must call task.configure(embedding) first"

        u.validate(dsets, "dsets", {
            "train": tft.DatasetSpec({
                "values": tf.TensorSpec([None, None, self.ds_cfg.n_input_dims], u.dtype()),
                "seq_idxs": tf.TensorSpec([None, None, len(self.ds_cfg.seq_dims)], tf.int32),
                # "extra": tft.NoneTensorSpec(),
            }),
            "val": tft.DatasetSpec({
                "values": tf.TensorSpec([None, None, self.ds_cfg.n_input_dims], u.dtype()),
                "seq_idxs": tf.TensorSpec([None, None, len(self.ds_cfg.seq_dims)], tf.int32),
                # "extra": tft.NoneTensorSpec(),
            }),
            "test": tft.DatasetSpec({
                "values": tf.TensorSpec([None, None, self.ds_cfg.n_input_dims], u.dtype()),
                "seq_idxs": tf.TensorSpec([None, None, len(self.ds_cfg.seq_dims)], tf.int32),
                # "extra": {} # optional keys here
            }),
        })
        seq_len = dsets.train.element_spec["values"].shape[1]

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
        if seq_len != self.chunk_size:

            get_chunk = make_get_chunk_batched_ragged(self.chunk_size)

            def do_chunk(x):

                data, seq_idxs = get_chunk([
                    x["values"],
                    x["seq_idxs"],
                ])

                return {
                    **x,
                    "values": data,
                    "seq_idxs": seq_idxs,
                }

            # chunk
            dsets = dsets.map(do_chunk)
        else:
            print("Not chunking seqs")

        dsets = dsets.map(lambda x: {
            **x,
            "values": tf.ensure_shape(x["values"], [None, self.chunk_size, self.ds_cfg.n_input_dims]),
            "seq_idxs": tf.ensure_shape(x["seq_idxs"], [None, self.chunk_size, len(self.ds_cfg.seq_dims)]),
        })

        dsets = dsets.map(lambda x: {
            "inputs": {
                "values": x["values"][:, :-1, :],
                "seq_idxs": x["seq_idxs"][:, :-1, :],
                "target_idxs": x["seq_idxs"],
            },
            "targets": x["values"],
            "extra": x["extra"],
        })

        # dsets.map(lambda x: dbg(x, "pre-embed"))

        # dbg(dsets)

        return dsets

    def make_final_layer(self) -> tf.keras.layers.Layer:

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"
        assert self.model_cfg is not None, "Must call task.configure(embedding) first"

        inputs = u.input_dict(
            Input([None, self.model_cfg.n_output_embd], name="embd"),
        )

        dense = tf.keras.layers.Dense(self.ds_cfg.n_input_dims, name="dense", kernel_regularizer=u.reg())

        def call(inputs):
            embd = inputs["embd"]
            outputs = dense(embd)
            if u.is_debug():
                tf.cond(
                    tf.reduce_any(tf.math.is_nan(outputs)),
                    lambda: tf.print(f"WARNING: NaNs in outputs of {self.identifier}."),
                    lambda: tf.print(f"WARNING: No NaNs in outputs of {self.identifier}."),
                )
            return outputs

        return Model(
            inputs=inputs,
            outputs=call(inputs),
            name="outputs",
        )

    def make_loss_fn(self) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        "Mean-squared error loss function"


        def loss_fn(targets, inputs):

            outputs = inputs["values"]

            return tf.reduce_mean(tf.square(targets - outputs))

        inputs = (
            Input([None, self.ds_cfg.n_input_dims], name="targets"),
            u.input_dict(
                Input([None, self.ds_cfg.n_input_dims], name="values"),
            ),
        )
        return Model(inputs=inputs, outputs=loss_fn(*inputs), name="loss_fn")

    def make_predict_fn(self, model):
        """
        Build a function to predict the next vector in the sequence, starting with optional seed data.
        """

        assert self.adapt_in is not None, "Must call task.configure(embedding) first"

        @tf.function
        def predict_fn(seed_inputs, seq_idxs, out_var: tf.Variable):
            batch_size = tf.shape(seed_inputs)[0]
            seed_len = tf.shape(seed_inputs)[1]

            tf.assert_equal(tf.shape(seq_idxs)[0], batch_size, f"idxs must have the same batch size as seed_inputs.")

            out_var[:, :seed_len, :].assign(seed_inputs)

            n = out_var.shape[1]
            for i in tf.range(seed_len, n): # converted to tf.while_loop
                inputs = {
                    "values": out_var[:, :i],
                    "seq_idxs": seq_idxs[:, :i],
                }
                outputs = model(inputs, training=False)
                out_var[:, i, :].assign(outputs[:, -1, :])
            return out_var

        warned = False
        def predict_wrapper(inputs, seed_len = self.pred_seed_len, output_len = self.pred_output_len):
            nonlocal warned

            u.validate(inputs, "inputs", {
                "values": tf.TensorSpec([None, None, self.ds_cfg.n_input_dims], u.dtype()),
                "seq_idxs": tf.TensorSpec([None, None, len(self.ds_cfg.seq_dims)], tf.int32),
            })
            data = inputs["values"]
            seq_idxs = inputs["seq_idxs"]
            batch_size = shape(data)[0]
            seq_len = shape(data)[1]
            n_features = shape(data)[2]

            assert seed_len <= seq_len,   f"seed_len must be less than or equal to the sequence length. Got {seed_len} and {seq_len}"
            assert output_len > 0,        f"output_len must be greater than 0. Got {output_len}"
            assert seed_len < output_len, f"seed_len must be less than output_len. Got {seed_len} and {output_len}"


            if output_len > self.chunk_size and not warned:
                print(f"WARNING: pred_output_len should be less than or equal to chunk_size. This is because the model has not been trained on longer sequences. Got pred_output_len={output_len} and chunk_size={self.chunk_size}", file=sys.stderr)
                warned = True

            seed_input = data[:, :seed_len, :]

            out_var = tf.Variable(tf.zeros([batch_size, output_len, n_features], u.dtype()))
            predict_fn(seed_input, seq_idxs, out_var)
            if u.is_debug() and tf.reduce_any(tf.math.is_nan(out_var)):
                tf.print(f"WARNING: NaNs in outputs of {self.identifier}.")
            return {
                "values": out_var,
            }

        return predict_wrapper


@export
@dataclass
class VectorSequenceAngleMSE(Task):
    """
    Predict the next vector in a sequence, as a vector of unit vectors.
    Because the outputs are (many-dimensional) vectors, this is a regression
    task only.

    For a distribution prediction task, use NextTokenPrediction which
    predicts a categorical distribution over a vocabulary of vectors.
    """

    def __init__(self, chunk_size: int, pred_seed_len: int = 0, pred_output_len: int = None, n_test_val_repeats: int = 100):
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

        self.pred_seed_len: int = pred_seed_len
        """
        Default amount of seed data to use when predicting output sequences.
        """

        self.pred_output_len: int = chunk_size if pred_output_len is None else pred_output_len
        """
        Default length of predicted sequences.
        """

        assert self.pred_seed_len < self.pred_output_len, f"pred_seed_len must be less than pred_output_len. Got pred_seed_len={self.pred_seed_len} and pred_output_len={self.pred_output_len}"


    def configure(self, embedding: MxEmbedding):

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"

        n_input_dims = self.ds_cfg.n_input_dims
        sequence_length = self.chunk_size

        ### Set embedding down_cfg ###
        if isinstance(embedding, AngleVectorSequence):
            embedding.receive_task_config(embedding.task_config_type(
                n_input_dims,
                sequence_length,
            ))
            def adapt_in(x):
                return {
                    **x,
                    "inputs": {
                        "angles": x["inputs"]["input"],
                        "seq_idxs": x["inputs"]["seq_idxs"],
                    }
                }
            self.adapt_in = adapt_in
        else:
            raise NotImplementedError(f"Task {type_name(self)} does not support Embedding {type_name(embedding)}. If using autoreload in IPython, try restarting the interpreter.")

    def process(self, dsets: DSets) -> DSets:

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"
        assert self.adapt_in is not None, "self.adaptor was not set by task.configure(embedding)"

        u.validate(dsets, "dsets", {
            "train": {
                "data": tf.TensorSpec([None, None], u.dtype()),
                "seq_idxs": tf.TensorSpec([None], tf.int32),
            },
            "val": {
                "data": tf.TensorSpec([None, None], u.dtype()),
                "seq_idxs": tf.TensorSpec([None], tf.int32),
            },
            "test": {
                "data": tf.TensorSpec([None, None], u.dtype()),
                "seq_idxs": tf.TensorSpec([None], tf.int32),
            },
        })

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

        get_chunk = make_get_chunk(self.chunk_size)

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
                "seq_idxs": x["seq_idxs"][:-1],
                "target_idxs": x["seq_idxs"],
            },
            "targets": x["data"],
            "extra": x["extra"],
        })

        return dsets

    def make_final_layer(self) -> tf.keras.layers.Layer:

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"
        assert self.model_cfg is not None, "Must call task.configure(embedding) first"

        inputs = u.input_dict(
            Input([None, self.model_cfg.n_output_embd], name="embd"),
        )
        dense = tf.keras.layers.Dense(self.ds_cfg.n_input_dims * 2, kernel_regularizer=u.reg())
        def call(inputs):
            outputs = dense(inputs["embd"])
            outputs = ein.rearrange(outputs, "... seq (feat sincos) -> ... seq feat sincos", sincos=2)
            if u.is_debug():
                tf.cond(
                    tf.reduce_any(tf.math.is_nan(outputs)),
                    lambda: tf.print(f"WARNING: NaNs in outputs of {self.identifier}."),
                    lambda: tf.print(f"WARNING: No NaNs in outputs of {self.identifier}."),
                )
            return outputs

        return Model(
            inputs=inputs,
            outputs=call(inputs),
            name="outputs",
        )

    def make_loss_fn(self) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        "Angular mean-squared-error loss."

        @tf.function
        @u.tf_scope
        def angular_mse_loss(targets, outputs):
            "Angular mean-squared-error loss."

            unit_vectors = outputs["unit_vectors"]

            target_sin = tf.sin(targets)
            target_cos = tf.cos(targets)
            sin = unit_vectors[..., 0]
            cos = unit_vectors[..., 1]
            return tf.reduce_mean(tf.square(target_sin - sin) + tf.square(target_cos - cos))

        inputs = (
            Input([None, self.ds_cfg.n_input_dims], name="targets"),
            u.input_dict(
                Input([None, self.ds_cfg.n_input_dims, 2], name="unit_vectors"),
            ),
        )
        return Model(inputs=inputs, outputs=angular_mse_loss(**inputs), name="loss_fn")

    def make_predict_fn(self, model):
        """
        Build a function to predict the next vector in the sequence, starting with optional seed data.
        """

        assert self.adapt_in is not None, "Must call task.configure(embedding) first"

        def unit_vector_to_angle(unit_vector):
            return tf.math.atan2(unit_vector[..., 0], unit_vector[..., 1])

        @tf.function
        def predict_fn(seed_inputs, idxs, out_var: tf.Variable):
            batch_size = tf.shape(seed_inputs)[0]
            seed_len = tf.shape(seed_inputs)[1]

            tf.assert_equal(tf.shape(idxs)[0], batch_size, f"idxs must have the same batch size as seed_inputs.")

            out_var[:, :seed_len, :].assign(seed_inputs)

            n = out_var.shape[1]
            for i in tf.range(seed_len, n): # converted to tf.while_loop
                inputs = {
                    "angles": out_var[:, :i, :],
                    "seq_idxs": idxs[:, :i],
                }
                outputs = model(inputs, training=False)
                outputs = unit_vector_to_angle(outputs[:, -1, :, :])
                out_var[:, i, :].assign(outputs)
            return out_var

        warned = False
        def predict_wrapper(inputs, seed_len = self.pred_seed_len, output_len = self.pred_output_len):
            nonlocal warned

            # new style validation using u.validate

            u.validate(inputs, "inputs", {
                "data": tf.TensorSpec([None, None, self.ds_cfg.n_input_dims], u.dtype()),
                "seq_idxs": tf.TensorSpec([None, None], tf.int32),
            })
            data = inputs["data"]
            seq_idxs = inputs["seq_idxs"]
            batch_size = shape(data)[0]
            seq_len = shape(data)[1]

            assert seed_len <= seq_len, f"seed_len ({seed_len}) must be <= seq_len ({seq_len})"
            assert output_len > 0, f"output_len ({output_len}) must be > 0"


            if output_len > self.chunk_size and not warned:
                print(f"WARNING: pred_output_len should be less than or equal to chunk_size. This is because the model has not been trained on longer sequences. Got pred_output_len={output_len} and chunk_size={self.chunk_size}", file=sys.stderr)
                warned = True

            seed_input = data[:, :seed_len, :]

            n_features = data.shape[2]

            out_var = tf.Variable(tf.zeros([batch_size, output_len, n_features]))
            predict_fn(seed_input, seq_idxs, out_var)
            if u.is_debug() and tf.reduce_any(tf.math.is_nan(out_var)).numpy():
                raise ValueError(f"NaNs in output of predict_fn of {self.identifier}")

            return {
                "angles": out_var,
            }

        return predict_wrapper


if __name__ == '__main__':
    u.set_debug(True)
    task = VectorSequenceMSE(
        chunk_size=13,
    )
    embedding = VectorSinusoidalMultidim(
        n_embd=128,
    )

    task.recieve_dataset_config(task.ds_config_type(
        seq_dims=[100, 100],
        n_input_dims=3,
        already_batched=False,
    ))

    task.configure(embedding)

    data = Dataset.from_tensor_slices((
        tf.random.uniform([5000, 100, 100, 3], dtype=tf.float32),
        tf.random.uniform([5000], minval=0, maxval=13, dtype=tf.int32),
    ))

    data = data.map(lambda x, y: {
        "values": ein.rearrange(x, "h w c -> (h w) c"),
        "labels": y,
        "seq_idxs": u.multidim_indices(shape(x)[:2]),
        "extra": None,
    })

    data = DSets(
        train=data.take(4000),
        val=data.skip(4000).take(500),
        test=data.skip(4500),
    )
    data = data.batch(5, 5)

    dbg(data, "task input")
    data = dbg(task.process(data), "task output")

    dbg(next(iter(data.train)), "first item")
