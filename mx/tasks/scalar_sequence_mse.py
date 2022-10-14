
from mx.embedding import TransformerMultidimScalar
from mx.prelude import *
from mx.pipeline import Embedding, Task
from mx.utils import DSets
from mx.tasks.utils import make_get_chunk


@export
@dataclass
class ScalarSequenceMSE(Task):
    """
    Takes a sequence of scalars and their N-dimensional indices,
    cuts them into chunks of length `chunk_size`, and yields an infinite
    stream of chunks.

    Mean-squared-error loss.
    """

    def __init__(self, chunk_size: int, pred_seed_len: int = 0, pred_output_len: int = None, n_test_val_repeats: int = 100):
        super().__init__(
            name = "Scalar Sequence MSE",
            identifier = "scalar-sequence-mse",
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


    def configure(self, embedding: Embedding):

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"

        n_input_dims = self.ds_cfg.n_input_dims
        sequence_length = self.chunk_size

        ### Set embedding down_cfg ###
        if isinstance(embedding, TransformerMultidimScalar):
            embedding.receive_task_config(embedding.task_config_type(
                n_input_dims,
                sequence_length,
            ))
            def adapt_in(x):
                return {
                    **x,
                    "inputs": {
                        "values": x["inputs"]["input"],
                        "input_idxs": x["inputs"]["input_idxs"],
                    }
                }
            self.adapt_in = adapt_in
        else:
            raise NotImplementedError(f"Embedding type {type_name(embedding)} not implemented")

    def process(self, dsets: DSets) -> DSets:

        assert self.ds_cfg is not None, "Must call dataset.configure(task) first"
        assert self.adapt_in is not None, "Must call task.configure(embedding) first"

        element_spec = dsets.train.element_spec

        assert type(element_spec) is dict, f"{self.identifier} requires a dataset of dicts"

        for k in element_spec.keys():
            assert k in dsets.test.element_spec, f"Test set was different from train set: {k} was missing"
            assert k in dsets.val.element_spec, f"Val set was different from train set: {k} was missing"

        assert "values" in element_spec, f"{self.identifier} requires a dataset with a 'data' key"
        assert len(element_spec["values"].shape) == 2, f"Data for {self.identifier} must have only a single sequence dimension, and a single feature dimension. Got shape {element_spec['data'].shape}"

        assert "seq_idxs" in element_spec, f"{self.identifier} requires a dataset with a 'seq_idxs' key"
        assert len(element_spec["seq_idxs"].shape) == 1, f"seq_idxs for {self.identifier} must have shape [seq]. Got shape {element_spec['seq_idxs'].shape}"

        assert element_spec["values"].shape[0] == element_spec["seq_idxs"].shape[0], f"Data and seq_idxs must have the same sequence length. Got {element_spec['data'].shape[0]} ≠ {element_spec['seq_idxs'].shape[0]}"

        assert dsets.test.element_spec["values"] == element_spec["values"], "Test set was different from train set: data shape was different"
        assert dsets.test.element_spec["seq_idxs"] == element_spec["seq_idxs"], "Test set was different from train set: seq_idxs shape was different"
        assert dsets.val.element_spec["values"] == element_spec["values"], "Val set was different from train set: data shape was different"
        assert dsets.val.element_spec["seq_idxs"] == element_spec["seq_idxs"], "Val set was different from train set: seq_idxs shape was different"

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

        dsets = dsets.map(lambda x: {
            "inputs": {
                "input": x["values"][:-1],
                "input_idxs": x["seq_idxs"][:-1],
                "target_idxs": x["seq_idxs"],
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
        @u.tf_scope
        def angular_mse_loss(targets, outputs):
            "Angular mean-squared-error loss."

            unit_vectors = outputs["unit_vectors"]

            target_sin = tf.sin(targets)
            target_cos = tf.cos(targets)
            sin = unit_vectors[..., 0]
            cos = unit_vectors[..., 1]
            return tf.reduce_mean(tf.square(target_sin - sin) + tf.square(target_cos - cos))

        return angular_mse_loss

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
                    "input_idxs": idxs[:, :i],
                }
                outputs = model(inputs, training=False)
                out = outputs["unit_vectors"]
                out = unit_vector_to_angle(out[:, -1, :, :])
                out_var[:, i, :].assign(out)
            return out_var

        def predict_wrapper(inputs, seed_len = self.pred_seed_len, output_len = self.pred_output_len):


            assert isinstance(inputs, dict),           f"inputs must be a dict. Got {type_name(inputs)}"

            assert "values" in inputs,                   f"inputs must contain key 'data'. Got {inputs.keys()}"
            data = inputs["values"]
            assert tf.is_tensor(data),                 f"inputs['data'] must be a tensor. Got {type_name(inputs['data'])}"
            assert len(data.shape) in [2, 3],          f"inputs['data'] must be a 2D or 3D tensor. Got {inputs['data'].shape}"

            if len(data.shape) == 2:
                data = tf.expand_dims(data, 0)

            assert data.dtype == tf.float32,           f"inputs['data'] must be a float32 tensor. Got {inputs['data'].dtype}"

            assert "seq_idxs" in inputs,               f"inputs must contain key 'seq_idxs'. Got {inputs.keys()}"
            seq_idxs = inputs["seq_idxs"]
            assert tf.is_tensor(seq_idxs),             f"inputs['seq_idxs'] must be a tensor. Got {type_name(inputs['seq_idxs'])}"
            assert len(seq_idxs.shape) in [1, 2],      f"inputs['seq_idxs'] must be a 2D tensor. Got {inputs['seq_idxs'].shape}"

            if len(seq_idxs.shape) == 1:
                seq_idxs = tf.expand_dims(seq_idxs, 0)

            assert seq_idxs.dtype == tf.int32,         f"inputs['seq_idxs'] must be a int32 tensor. Got {inputs['seq_idxs'].dtype}"

            assert data.shape[0] == seq_idxs.shape[0], f"inputs['data'] and inputs['seq_idxs'] must have the same batch size. Got {inputs['data'].shape[0]} and {inputs['seq_idxs'].shape[0]}"
            batch_size = data.shape[0]
            assert data.shape[1] == seq_idxs.shape[1], f"inputs['data'] and inputs['seq_idxs'] must have the same sequence length. Got {inputs['data'].shape[1]} and {inputs['seq_idxs'].shape[1]}"
            seq_len = data.shape[1]
            assert seed_len <= seq_len,                f"seed_len must be less than or equal to the sequence length. Got {seed_len} and {seq_len}"
            assert output_len > 0,                     f"output_len must be greater than 0. Got {output_len}"
            assert seed_len < output_len,              f"seed_len must be less than output_len. Got {seed_len} and {output_len}"

            if output_len > self.chunk_size:
                print(f"WARNING: pred_output_len should be less than or equal to chunk_size. This is because the model has not been trained on longer sequences. Got pred_output_len={output_len} and chunk_size={self.chunk_size}", file=sys.stderr)

            seed_input = data[:, :seed_len, :]

            n_features = data.shape[2]

            out_var = tf.Variable(tf.zeros([batch_size, output_len, n_features]))
            predict_fn(seed_input, seq_idxs, out_var)
            return {
                "angles": out_var,
            }

        return predict_wrapper