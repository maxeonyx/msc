from mx.embeddings import DebugCodebook, DebugCodebookTriples, Multidim_TaskConfig, VectorCodebookMultidim
import mx.predict as pred
from mx.prelude import *
from mx.progress import Progress, create_progress_manager

from mx.utils import DSets
from mx.tasks.utils import make_get_chunk_batched_ragged, make_get_random_slices_batched_ragged
from mx.embeddings import AngleCodebook
from mx.pipeline import Task, MxEmbedding, Task_DatasetConfig


@export
@dataclass
class MultidimTask_DatasetConfig(Task_DatasetConfig):
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


    def __init__(
        self,
        chunk_size: int,
        pred_seed_len: int = 0,
        pred_output_len: int = None,
        n_test_val_repeats: int = 100,
        name = "vecseq",
        desc = "Vector Sequence (MSE loss)",
    ):
        super().__init__(
            name=name,
            desc=desc,
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

        self.ds_config_cls: Type[MultidimTask_DatasetConfig] = MultidimTask_DatasetConfig
        "Required dataset-specific config"

        self.ds_cfg: MultidimTask_DatasetConfig = None

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
                    lambda: tf.print(f"WARNING: NaNs in outputs of {self.name}."),
                    lambda: tf.print(f"WARNING: No NaNs in outputs of {self.name}."),
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
        def predict_wrapper(inputs, seed_len = self.pred_seed_len, output_len = self.pred_output_len, pm:Progress=None):
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
                tf.print(f"WARNING: NaNs in outputs of {self.name}.")
            return {
                "values": out_var,
            }

        return predict_wrapper



@export
@dataclass
class RandomSequenceMSE(Task):
    """
    Takes a sequence of vectors and their N-dimensional indices,
    takes `n_slices` unique samples at random from the sequence dimension,
    and yields an infinite stream of chunks.

    Mean-squared-error loss.
    """


    def __init__(
        self,
        n_slices: int,
        pred_seed_len: int = 0,
        pred_output_len: int = None,
        n_test_val_repeats: int = 100,
        name = "randseq",
        desc = "Random Sequence (MSE loss)",
    ):
        super().__init__(
            name=name,
            desc=desc,
        )

        self.n_slices: int = n_slices
        "Length of chunks (sequence length)"

        self.n_test_val_repeats: int = n_test_val_repeats
        """
        Number of slices to take out of each example to make validation and testing data.
        In training, it's infinite and the number depends on the number of training steps
        and batch size.
        """

        self.pred_seed_len: int = pred_seed_len
        """
        Default amount of seed data to use when predicting output sequences.
        """

        self.pred_output_len: int = n_slices if pred_output_len is None else pred_output_len
        """
        Default length of predicted sequences.
        """

        ## set by dataset.configure(task) ##
        self.ds_config_cls: Type[MultidimTask_DatasetConfig] = MultidimTask_DatasetConfig
        "Required dataset-specific config"

        self.ds_cfg: MultidimTask_DatasetConfig = None

        # self.model_config_type: Type[Task.ModelSpecificConfig] = Task.ModelSpecificConfig
        # "Required model-specific config"

        assert self.pred_seed_len < self.pred_output_len, f"pred_seed_len must be less than pred_output_len. Got pred_seed_len={self.pred_seed_len} and pred_output_len={self.pred_output_len}"


    def configure(self, embedding: MxEmbedding):

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"

        ### Set embedding down_cfg ###
        if isinstance(embedding, DebugCodebookTriples):
            embedding.receive_task_config(embedding.task_config_type(
                n_input_dims=self.ds_cfg.n_input_dims,
                seq_dims=self.ds_cfg.seq_dims,
            ))
            def adapt_in(x):
                return {
                    **x,
                    "inputs": {
                        "values": x["inputs"]["values"],
                        "inp_idxs": x["inputs"]["inp_idxs"],
                        "tar_idxs": x["inputs"]["tar_idxs"],
                    }
                }
            self.adapt_in = adapt_in
        else:
            raise NotImplementedError(f"Task {type_name(self)} does not support Embedding {type_name(embedding)}. If using autoreload in IPython, try restarting the interpreter.")

    def process(self, dsets: DSets) -> DSets:

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"
        assert self.adapt_in is not None, "Must call task.configure(embedding) first"

        ds_spec = tft.DatasetSpec(element_spec={
            "values": tf.TensorSpec([None, None, self.ds_cfg.n_input_dims], u.dtype()),
            "seq_idxs": tf.TensorSpec([None, None, len(self.ds_cfg.seq_dims)], tf.int32),
            # "extra": tft.NoneTensorSpec(),
        })

        u.validate(dsets, "dsets", {
            "train": ds_spec,
            "val": ds_spec,
            "test": ds_spec,
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

        get_slices = make_get_random_slices_batched_ragged(self.n_slices)

        def do_get_slices(x):

            data, seq_idxs = get_slices([
                x["values"],
                x["seq_idxs"],
            ])

            return {
                **x,
                "values": data,
                "seq_idxs": seq_idxs,
            }

        # chunk
        dsets = dsets.map(do_get_slices)

        dsets = dsets.map(lambda x: {
            **x,
            "values": tf.ensure_shape(x["values"], [None, self.n_slices, self.ds_cfg.n_input_dims]),
            "seq_idxs": tf.ensure_shape(x["seq_idxs"], [None, self.n_slices, len(self.ds_cfg.seq_dims)]),
        })

        dsets = dsets.map(lambda x: {
            "inputs": {
                "values": x["values"][:, :-1, :],
                "inp_idxs": x["seq_idxs"][:, :-1, :],
                "tar_idxs": x["seq_idxs"],
            },
            "targets": x["values"],
            "extra": x["extra"],
        })

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
                    lambda: tf.print(f"WARNING: NaNs in outputs of {self.name}."),
                    lambda: tf.print(f"WARNING: No NaNs in outputs of {self.name}."),
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

        # @tf.function
        def predict_fn(seed_inputs, seq_idxs, out_var: tf.Variable):
            batch_size = tf.shape(seed_inputs)[0]
            seed_len = tf.shape(seed_inputs)[1]

            tf.assert_equal(tf.shape(seq_idxs)[0], batch_size, f"idxs must have the same batch size as seed_inputs.")

            if seed_len > 0:
                out_var[:, :seed_len, :].assign(seed_inputs)

            n = out_var.shape[1]
            for i in tf.range(seed_len, n): # converted to tf.while_loop
                inputs = {
                    "values": out_var[:, :i],
                    "inp_idxs": seq_idxs[:, :i],
                    "tar_idxs": seq_idxs[:, :i+1],
                }
                outputs = model(inputs, training=False)
                out_var[:, i, :].assign(outputs[:, -1, :])
            return out_var

        warned = False
        def predict_wrapper(inputs, seed_len = self.pred_seed_len, output_len = self.pred_output_len, pm:Progress=None):
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


            if output_len > self.n_slices and not warned:
                print(f"WARNING: pred_output_len should be less than or equal to chunk_size. This is because the model has not been trained on longer sequences. Got pred_output_len={output_len} and chunk_size={self.n_slices}", file=sys.stderr)
                warned = True

            seed_input = data[:, :seed_len, :]

            out_var = tf.Variable(tf.zeros([batch_size, output_len, n_features], u.dtype()))

            dbg(seed_input, "seed_input")
            dbg(seq_idxs, "seq_idxs")
            dbg(out_var, "out_var")

            predict_fn(seed_input, seq_idxs, out_var)
            if u.is_debug() and tf.reduce_any(tf.math.is_nan(out_var)):
                tf.print(f"WARNING: NaNs in outputs of {self.name}.")
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

    def __init__(
        self,
        chunk_size: int,
        pred_seed_len: int = 0,
        pred_output_len: int = None,
        n_test_val_repeats: int = 100,
        name = "anglevec",
        desc = "Angle Vector Prediction (single sequence dim only)",
    ):
        super().__init__(
            name=name,
            desc=desc,
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
        self.ds_config_cls: Type[Task_DatasetConfig] = Task_DatasetConfig

        assert self.pred_seed_len < self.pred_output_len, f"pred_seed_len must be less than pred_output_len. Got pred_seed_len={self.pred_seed_len} and pred_output_len={self.pred_output_len}"


    def configure(self, embedding: MxEmbedding):

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"

        n_input_dims = self.ds_cfg.n_input_dims
        sequence_length = self.chunk_size

        ### Set embedding down_cfg ###
        if isinstance(embedding, AngleCodebook):
            embedding.receive_task_config(embedding.task_config_type(
                n_input_dims,
                sequence_length,
            ))
            def adapt_in(x):
                return {
                    **x,
                    "inputs": {
                        "context/values": x["inputs"]["values"],
                        "context/inp_idxs": x["inputs"]["inp_idxs"],
                    }
                }
            self.adapt_in = adapt_in
        else:
            raise NotImplementedError(f"Task {type_name(self)} does not support Embedding {type_name(embedding)}. If using autoreload in IPython, try restarting the interpreter.")

    def process(self, dsets: DSets) -> DSets:

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"
        assert self.adapt_in is not None, "self.adaptor was not set by task.configure(embedding)"

        ds_spec = tft.DatasetSpec(element_spec={
            "angles": tf.RaggedTensorSpec([None, None, self.ds_cfg.n_input_dims], u.dtype(), ragged_rank=1),
            "seq_idxs": tf.RaggedTensorSpec([None, None, 1], tf.int32, ragged_rank=1),
        })

        u.validate(dsets, "dsets", {
            "train": ds_spec,
            "val": ds_spec,
            "test": ds_spec,
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

        get_chunk = make_get_chunk_batched_ragged(self.chunk_size)

        def do_chunk(x):

            angles, seq_idxs = get_chunk([
                x["angles"],
                x["seq_idxs"],
            ])

            return {
                **x,
                "angles": angles,
                "seq_idxs": seq_idxs,
            }

        # chunk
        dsets = dsets.map(do_chunk)

        dsets = dsets.map(lambda x: {
            "inputs": {
                "values": x["angles"][:, :-1],
                "inp_idxs": x["seq_idxs"][:, :-1],
                "tar_idxs": x["seq_idxs"],
            },
            "targets": x["angles"],
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
                    lambda: tf.print(f"WARNING: NaNs in outputs of {self.name}."),
                    lambda: tf.print(f"WARNING: No NaNs in outputs of {self.name}."),
                )
            return outputs

        return Model(
            inputs=inputs,
            outputs=call(inputs),
            name="outputs",
        )

    def make_loss_fn(self) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        "Angular mean-squared-error loss."

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
        return Model(inputs=inputs, outputs=angular_mse_loss(*inputs), name="loss_fn")

    def make_predict_fn(self, model):
        """
        Build a function to predict the next vector in the sequence, starting with optional seed data.
        """

        assert self.adapt_in is not None, "Must call task.configure(embedding) first"

        def unit_vector_to_angle(unit_vector):
            return tf.math.atan2(unit_vector[..., 0], unit_vector[..., 1])

        def predict_wrapper(inputs, seed_len = self.pred_seed_len, output_len = self.pred_output_len, pm:Progress=None):

            tp(inputs, "inputs")

            # unrag
            inputs = {
                "angles": tf.gather(inputs["angles"], tf.range(output_len), axis=1),
                "seq_idxs": tf.gather(inputs["seq_idxs"], tf.range(output_len), axis=1),
            }

            u.validate(inputs, "inputs", {
                "angles": tf.TensorSpec([None, None, self.ds_cfg.n_input_dims], u.dtype()),
                "seq_idxs": tf.TensorSpec([None, None, 1], tf.int32),
            })

            batch_size = shape(inputs["angles"])[0]
            seq_len = shape(inputs["angles"])[1]

            assert shape(inputs["seq_idxs"])[0] == batch_size, f"inputs['seq_idxs'] must have same batch size as inputs['angles'], but got inputs['seq_idxs'].shape[0] = {shape(inputs['seq_idxs'])[0]} and inputs['angles'].shape[0] = {batch_size}"

            assert seed_len <= seq_len, f"seed_len ({seed_len}) must be <= seq_len ({seq_len})"
            assert output_len > 0, f"output_len ({output_len}) must be > 0"


            if seed_len > 0:
                seed = inputs["angles"][:, :seed_len]
            else:
                seed = None

            if output_len > self.chunk_size and not warned:
                print(f"WARNING: pred_output_len should be less than or equal to chunk_size. This is because the model has not been trained on longer sequences. Got pred_output_len={output_len} and chunk_size={self.chunk_size}", file=sys.stderr)
                warned = True

            def output_fn(angles, dist):
                angles = u.angle_wrap(angles)
                angles = angles[..., None]
                colors = u.colorize(angles, vmin=-pi, vmax=pi, cmap="twilight_shifted")
                return ein.rearrange(
                    colors,
                    '... feat chan -> ... (feat chan)',
                )

            return pred.predict(
                name="Predict angles",
                desc="Predicting angle vectors...",
                cfg=pred.PredictInputs(
                    out_seq_shape=[seq_len],
                    out_feat_shape=[self.ds_cfg.n_input_dims*3],
                    model=model,
                    model_supports_querying=False,
                    model_outputs_distribution=False,
                    sampling_order="fixed",
                    input_data=seed,
                    idxs=inputs["seq_idxs"],
                ),
                pm=pm,
                data_out_fn=output_fn,
                default_out_var_val=ein.repeat(
                    pred.DEFAULT_BG_COLOR,
                    'chan -> (feat chan)',
                    feat=self.ds_cfg.n_input_dims,
                ),
                outp_to_inp_fn=unit_vector_to_angle,
            )


        return predict_wrapper


if __name__ == '__main__':
    u.set_debug()

    data = Dataset.from_tensor_slices({
        "values": tf.random.uniform([5000, 100, 100, 3], dtype=tf.float32),
        "labels": tf.random.uniform([5000], minval=0, maxval=13, dtype=tf.int32),
    })
    data = data.map(lambda x: {
        "values": ein.rearrange(x["values"], "h w c -> (h w) c"),
        "labels": x["labels"],
        "seq_idxs": u.multidim_indices([100, 100]),
        "extra": None,
    })
    data = DSets(
        train=data.take(4000),
        val=data.skip(4000).take(500),
        test=data.skip(4500),
    )
    data = data.batch(5, 5)

    cfgs = [
        Box(
            name="chunks",
            task=VectorSequenceMSE(
                chunk_size=13,
            ),
            task_ds_cfg=MultidimTask_DatasetConfig(
                n_input_dims=3,
                seq_dims=[100, 100],
            ),
            embd=DebugCodebook(
                n_embd=10,
            ),
        ),
        Box(
            name="randomslices",
            task=RandomSequenceMSE(
                n_slices=13,
            ),
            task_ds_cfg=MultidimTask_DatasetConfig(
                n_input_dims=3,
                seq_dims=[100, 100],
            ),
            embd=DebugCodebookTriples(
                n_embd=10,
            ),
        ),
    ]

    for cfg in cfgs:
        task = cfg.task
        embd = cfg.embd
        name = cfg.name

        d = dbg(data, f"{name}: data before task.process")

        task.recieve_dataset_config(cfg.task_ds_cfg)
        task.configure(embd)

        d = dbg(task.process(d), f"{name}: data after task.process")

        d = d.train.map(lambda x: dbg(x, f"{name}: train datum"))

        print(f"Getting concrete data for task {task.name}...")
        next(iter(d))
        print(f"Done.")
        print()
        print()


    n_input_dims = 102
    seq_dims = [1000]
    data = Dataset.from_tensor_slices({
        "angles": tf.random.uniform([5000, *seq_dims, n_input_dims], dtype=tf.float32),
        "labels": tf.random.uniform([5000], minval=0, maxval=13, dtype=tf.int32),
    })
    data = data.map(lambda x: {
        "angles": x["angles"],
        "labels": x["labels"],
        "seq_idxs": u.multidim_indices(seq_dims),
        "extra": None,
    })
    data = DSets(
        train=data.take(4000),
        val=data.skip(4000).take(500),
        test=data.skip(4500),
    )
    data = data.batch(5, 5)
    cfg = Box(
        name="anglechunks",
        task=VectorSequenceAngleMSE(
            chunk_size=13,
        ),
        task_ds_cfg=Task_DatasetConfig(
            n_input_dims=n_input_dims,
        ),
        embd=AngleCodebook(
            n_embd=11,
            n_repeats=12,
        ),
    )
    task: VectorSequenceAngleMSE = cfg.task
    embd = cfg.embd
    name = cfg.name

    data = dbg(data, f"{name}: data before task.process")

    task.recieve_dataset_config(cfg.task_ds_cfg)
    task.configure(embd)

    d = dbg(task.process(data), f"{name}: data after task.process")

    d = d.train.map(lambda x: dbg(x, f"{name}: train datum"))

    print(f"Getting concrete data for task {task.name}...")
    x = next(iter(d))
    dbg(x, f"{name}: train datum")
    print(f"Done.")
    print()
    print()

    model_inputs = u.input_dict(
        Input([None, n_input_dims], dtype=u.dtype(), name="context/values"),
        Input([None, len(seq_dims)], dtype=tf.int32,  name="context/inp_idxs"),
    )

    def demo_angle_model(inputs):
        v = inputs["context/values"]
        batch_size = tf.shape(v)[0]
        seq_len = tf.shape(v)[1]
        n_feat = tf.shape(v)[2]

        v = tf.concat([
            tf.ones([batch_size, 1, n_feat], dtype=tf.float32),
            v,
        ], axis=1)
        return tf.stack([
            tf.sin(tf.reduce_mean(v, axis=1)[:, None, :] * 0.9),
            tf.cos(tf.reduce_mean(v, axis=1)[:, None, :] * 1.1),
        ], axis=-1)

    model = Model(
        inputs=model_inputs,
        outputs=demo_angle_model(model_inputs),
        name="test",
    )

    x = next(iter(data.test))

    pf = task.make_predict_fn(model)

    with create_progress_manager() as pm:
        tp(pf(x, pm=pm), "predicted angles & etc.")
