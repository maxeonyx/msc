from __future__ import annotations

from mx.prelude import *

@export
class PrependToken(u.MxModule):
    """
    Prepend a sentinel embedding to the beginning of the sequence.

    >>> n_embd = 1
    >>> initializer = keras.initializers.Constant(0.)
    >>> prepend_token = PrependToken(embeddings_initializer=initializer)
    >>> embd = tf.ones([1, 1, n_embd])
    >>> embd = prepend_token(embd)
    >>> embd.shape
    TensorShape([1, 2, 1])
    >>> print(embd.numpy())
    [[[0.]
      [1.]]]
    >>> n_embd = 4
    >>> prepend_token = PrependToken(embeddings_initializer=initializer)
    >>> embd = tf.ones([1, 1, n_embd])
    >>> embd = prepend_token(embd)
    >>> embd.shape
    TensorShape([1, 2, 4])
    >>> print(embd.numpy())
    [[[0. 0. 0. 0.]
      [1. 1. 1. 1.]]]
    """

    def __init__(
        self,
        name="prepend_token",
        desc="Prepend Token",
        embeddings_initializer=None,
        **kwargs,
    ):
        super().__init__(name, desc, **kwargs)
        self.embeddings_initializer = embeddings_initializer


    def build(self, input_shape):
        n_embd = input_shape[-1]
        initializer = self.embeddings_initializer or tf.keras.initializers.TruncatedNormal(stddev=1/sqrt(n_embd))
        self.token_embedding = layers.Embedding(1, n_embd, embeddings_initializer=initializer, name=f"{self.name}/token_embd")
        self.built = True

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])

    def call(self, inputs):
        embd = inputs

        batch_size = shape(embd)[0]
        tokens = tf.tile(
            tf.constant(0, tf.int32)[None, None],
            [batch_size, 1],
        )
        token_embd = self.token_embedding(tokens)
        embd = tf.concat([token_embd, embd], axis=1) # concat along first sequence dim
        return embd

@export
class MultidimPosEnc(u.MxModule):
    """
    Sinusoidal positional encoding.

    >>> pos_enc = MultidimPosEnc(4, n_seq_dims=1)
    >>> idxs = u.multidim_indices([10])
    >>> pos_enc(idxs).shape
    TensorShape([10, 4])

    >>> pos_enc = MultidimPosEnc(4, n_seq_dims=1)
    >>> idxs_with_batch = ein.repeat(u.multidim_indices([10]), '... -> b ...', b=3)
    >>> pos_enc(idxs_with_batch).shape
    TensorShape([3, 10, 4])

    >>> pos_enc = MultidimPosEnc(4, n_seq_dims=2, max_wavelengths=[10000, 10000])
    >>> idxs = u.multidim_indices([10, 10])
    >>> pos_enc(idxs).shape
    TensorShape([100, 4])
    """

    def __init__(
        self,
        n_embd,
        n_seq_dims,
        max_wavelengths=None,
        name="pos_enc",
        desc=None,
        **kwargs
    ):
        super().__init__(name=name, desc=desc, **kwargs)

        assert n_embd % (n_seq_dims*2) == 0, f"embd_dim must be divisible by 2 and by n_seq_dims to use positional encoding, got embd_dim={n_embd}, n_seq_dims={n_seq_dims}"

        self.n_embd = n_embd
        self.n_seq_dims = n_seq_dims
        if max_wavelengths is None:
            self.max_wavelengths = [10000] * n_seq_dims
        elif n_seq_dims == 1 and isinstance(max_wavelengths, int):
            self.max_wavelengths = [max_wavelengths]
        else:
            assert len(max_wavelengths) == n_seq_dims, f"max_wavelengths must have length n_seq_dims, got {len(max_wavelengths)} and {n_seq_dims}"
            self.max_wavelengths = max_wavelengths

    # based from the keras source code
    # https://github.com/keras-team/keras-nlp/blob/v0.3.0/keras_nlp/layers/sine_position_encoding.py#L21

    @u.tf_scope
    def pos_enc(self, idxs, n, w):
        position = tf.cast(idxs, u.dtype())
        min_freq = 1. / w
        timescales = tf.pow(
            min_freq,
            tf.range(n, dtype=tf.float32) / n
        )
        timescales = tf.cast(timescales, u.dtype())
        position = ein.rearrange(position, '... seq -> ... seq ()')
        timescales = ein.rearrange(timescales, '... embd -> ... () embd')
        angles = position * timescales
        # even indices are sine, odd are cosine
        cos_mask = tf.cast(tf.range(n) % 2, u.dtype())
        sin_mask = 1 - cos_mask
        # embedding shape is [seq_length, hidden_size]
        positional_encodings = (
            tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
        )

        # scale norm. because we use sin/cos we scale by 1/sqrt(D/2) instead of 1/sqrt(D)
        positional_encodings *= tf.cast(tf.math.sqrt(tf.cast(n // 2, u.dtype())), u.dtype())

        return positional_encodings

    @u.tf_scope
    def call(self, idxs):
        positional_encodings = [
            self.pos_enc(
                idxs[..., i],
                self.n_embd // self.n_seq_dims,
                self.max_wavelengths[i]
            )
            for i in range(self.n_seq_dims)
        ]

        idx_embd = tf.concat(positional_encodings, axis=-1)

        return idx_embd

@export
class AngleEmbd(u.MxModule):
    """
    Embed angles as unit vectors and pass through a dense layer.

    >>> embd = AngleEmbd(n_embd=4)
    >>> embd(tf.constant([[[0.], [1.]]])).shape
    TensorShape([1, 2, 4])
    """

    def __init__(
        self,
        n_embd: int,
        name="angleembd",
        desc=None,
    ):
        if desc is None:
            desc = f"AngleEmbd({n_embd}"
        super().__init__(name=name, desc=desc)
        self.n_embd = n_embd

    def build(self, input_shape):
        self.n_feat = input_shape[-1]
        self.n_repeats = max(1, self.n_embd // (self.n_feat*2))
        self.dense = layers.Dense(self.n_embd, name=f"{self.name}/dense")
        self.built = True

    @u.tf_scope
    def call(self, angles):

        angles = tf.stack(
            [
                tf.sin(angles),
                tf.cos(angles),
            ],
            axis=-1,
        )
        angles = ein.repeat(
            angles,
            "... feat sincos -> ... (feat rep sincos)",
            rep=self.n_repeats,
        )

        # flatten to "embd" dim
        angle_embd = self.dense(angles)

        angle_embd /= tf.sqrt(tf.cast(angles.shape[-1]//2, u.dtype()))

        return angle_embd


@export
class ScalarEmbd(u.MxModule):
    """
    Embed values as unit vectors and pass through a dense layer.

    >>> embd = ScalarEmbd(n_embd=4)
    >>> embd(tf.constant([[[0.], [1.]]])).shape
    TensorShape([1, 2, 4])
    """

    def __init__(
        self,
        n_embd: int,
        name="scalarembd",
        desc=None,
    ):
        if desc is None:
            desc = f"ScalarEmbd({n_embd})"
        super().__init__(name=name, desc=desc)
        self.n_embd = n_embd

    def build(self, input_shape):
        self.n_feat = input_shape[-1]
        self.n_repeats = max(1, self.n_embd // (self.n_feat*2))
        self.dense = layers.Dense(self.n_embd, name=f"{self.name}/dense")
        self.built = True

    @u.tf_scope
    def call(self, values):

        values = ein.repeat(
            values,
            "... feat -> ... (feat rep)",
            rep=self.n_repeats,
        )

        # flatten to "embd" dim
        value_embd = self.dense(values)

        value_embd /= tf.sqrt(tf.cast(self.n_embd, u.dtype()))

        return value_embd


class CodebookEmbd(u.MxModule):
    """
    Embed tokens with a codebook. Wrapper around keras Embedding.

    >>> embd = CodebookEmbd(n_embd=4, n_tokens=10)
    >>> embd(tf.constant([[0, 1, 2]])).shape
    TensorShape([1, 3, 4])
    """

    def __init__(
        self,
        n_embd: int,
        n_tokens: int,
        name="codebookembd",
        desc=None,
    ):
        if desc is None:
            desc = f"CodebookEmbd({n_embd}, {n_tokens})"
        super().__init__(name=name, desc=desc)
        self.n_embd = n_embd
        self.n_tokens = n_tokens

    def build(self, input_shape):
        self.embd = layers.Embedding(
            self.n_tokens,
            self.n_embd,
            name=f"{self.name}/embd",
        )
        self.built = True

    @u.tf_scope
    def call(self, tokens):
        return self.embd(tokens)


@export
class DecoderOnlyEmbedding(u.MxModule):
    """
    Embedding for decoder-only model. Takes val embedding and position embedding layers.

    >>> n_embd = 4
    >>> max_pos = 10
    >>> inps = {
    ...     'ctx_inp_vals': tf.constant([[[0.], [1.]]]), # angles
    ...     'ctx_inp_idxs': tf.constant([[[0], [1]]]),  # positions
    ... }
    >>> embd = DecoderOnlyEmbedding(
    ...     val_embedder=ScalarEmbd(n_embd),
    ...     inp_pos_embedders=[layers.Embedding(max_pos, n_embd)],
    ... )
    >>> embd(inps)['ctx_embd'].shape
    TensorShape([1, 3, 4])


    >>> embd = DecoderOnlyEmbedding(
    ...     val_embedder=ScalarEmbd(n_embd),
    ...     inp_pos_embedders=MultidimPosEnc(n_embd, n_seq_dims=1),
    ... )
    >>> embd(inps)['ctx_embd'].shape
    TensorShape([1, 3, 4])


    >>> embd = DecoderOnlyEmbedding(
    ...     val_embedder=AngleEmbd(n_embd),
    ...     inp_pos_embedders=[layers.Embedding(max_pos, n_embd)],
    ... )
    >>> embd(inps)['ctx_embd'].shape
    TensorShape([1, 3, 4])


    >>> inps = {
    ...     'ctx_inp_vals': tf.constant([[[0.], [1.]]]), # angles
    ...     'ctx_inp_idxs': tf.constant([[[0], [1]]]),  # positions
    ...     'ctx_tar_idxs': tf.constant([[[0], [1], [2]]]),  # positions
    ... }
    >>> embd = DecoderOnlyEmbedding(
    ...     val_embedder=AngleEmbd(n_embd),
    ...     inp_pos_embedders=[layers.Embedding(max_pos, n_embd)],
    ...     tar_pos_embedders=[layers.Embedding(max_pos, n_embd)],
    ... )
    >>> embd(inps)['ctx_embd'].shape
    TensorShape([1, 3, 4])


    >>> embd = DecoderOnlyEmbedding(
    ...     val_embedder=ScalarEmbd(n_embd),
    ...     inp_pos_embedders=MultidimPosEnc(n_embd, n_seq_dims=1),
    ...     tar_pos_embedders=MultidimPosEnc(n_embd, n_seq_dims=1),
    ... )
    >>> embd(inps)['ctx_embd'].shape
    TensorShape([1, 3, 4])


    >>> inps = {
    ...     'ctx_inp_vals': tf.constant([[[0.], [1.]]]), # angles
    ...     'ctx_inp_idxs': tf.constant([[[0, 0], [1, 1]]]),  # positions
    ... }
    >>> embd = DecoderOnlyEmbedding(
    ...     val_embedder=AngleEmbd(n_embd),
    ...     inp_pos_embedders=[
    ...         layers.Embedding(max_pos, n_embd),
    ...         layers.Embedding(max_pos, n_embd),
    ...     ],
    ... )
    >>> embd(inps)['ctx_embd'].shape
    TensorShape([1, 3, 4])


    >>> inps = {
    ...     'ctx_inp_vals': tf.constant([[[0.], [1.]]]), # angles
    ...     'ctx_inp_idxs': tf.constant([[[0, 0], [1, 1]]]),  # positions
    ... }
    >>> embd = DecoderOnlyEmbedding(
    ...     val_embedder=AngleEmbd(n_embd),
    ...     inp_pos_embedders=MultidimPosEnc(n_embd, n_seq_dims=2),
    ... )
    >>> embd(inps)['ctx_embd'].shape
    TensorShape([1, 3, 4])
    """

    def __init__(
        self,
        val_embedder: u.MxModule,
        inp_pos_embedders: list[u.MxModule],
        tar_pos_embedders: list[u.MxModule] = None,
        begin_token_embeddings_initializer = None,
        name="decovalco",
        desc=None,
        **kwargs
    ):
        if desc is None:
            desc = f"Decoder only embd"
            if tar_pos_embedders is not None:
                desc += " (Trips)"
                name += "trips"
            else:
                desc += " (Pairs)"
                name += "pairs"
        super().__init__(name=name, desc=desc, **kwargs)

        self.val_embedder = val_embedder
        self.inp_pos_embedders = inp_pos_embedders
        self.tar_pos_embedders = tar_pos_embedders
        self.begin_token_embeddings_initializer = begin_token_embeddings_initializer

    def build(self, input_shape):
        assert 'ctx_inp_vals' in input_shape, f"Input must have 'ctx_inp_vals' key, got {input_shape}"
        assert 'ctx_inp_idxs' in input_shape, f"Input must have 'ctx_inp_idxs' key, got {input_shape}"
        if self.tar_pos_embedders is not None:
            assert 'ctx_tar_idxs' in input_shape, f"Input must have 'ctx_tar_idxs' key, got {input_shape}"

        self.batch_size = input_shape['ctx_inp_vals'][0]
        self.seq_len = input_shape['ctx_inp_vals'][1]
        self.n_feat = input_shape['ctx_inp_vals'][-1]
        self.n_idxs = input_shape['ctx_inp_idxs'][-1]
        self.n_embd = self.val_embedder.n_embd
        self.prepend_begin_token = PrependToken(embeddings_initializer=self.begin_token_embeddings_initializer, name=f"{self.name}/prepend_begin_token")
        self.prepend_begin_token.build([self.batch_size, self.seq_len, self.n_embd])

        self.val_embedder.build(input_shape['ctx_inp_vals'])
        if isinstance(self.inp_pos_embedders, list):
            for i, embd in enumerate(self.inp_pos_embedders):
                embd.build(input_shape['ctx_inp_idxs'])
        elif isinstance(self.inp_pos_embedders, u.MxModule):
            self.inp_pos_embedders.build(input_shape['ctx_inp_idxs'])
        else:
            raise ValueError(f"Unknown inp_pos_embedders type: {type(self.inp_pos_embedders)}")

        if self.tar_pos_embedders is not None:
            if isinstance(self.tar_pos_embedders, list):
                for i, embd in enumerate(self.tar_pos_embedders):
                    embd.build(input_shape['ctx_tar_idxs'])
            elif isinstance(self.tar_pos_embedders, u.MxModule):
                self.tar_pos_embedders.build(input_shape['ctx_tar_idxs'])
            else:
                raise ValueError(f"Unknown tar_pos_embedders type: {type(self.tar_pos_embedders)}")


    @u.tf_scope
    def call(self, inputs):
        assert 'ctx_inp_vals' in inputs, f"Input must have 'ctx_inp_vals' key, got {u.tf_str(inputs)}"
        assert 'ctx_inp_idxs' in inputs, f"Input must have 'ctx_inp_idxs' key, got {u.tf_str(inputs)}"
        if self.tar_pos_embedders is not None:
            assert 'ctx_tar_idxs' in inputs, f"Input must have 'ctx_tar_idxs' key, got {u.tf_str(inputs)}"

        vals = inputs['ctx_inp_vals']
        val_embd = self.val_embedder(vals)

        inp_idxs = inputs['ctx_inp_idxs']

        if isinstance(self.inp_pos_embedders, list):
            inp_pos_embd = tf.add_n([
                embedder(inp_idxs[..., i])
                for i, embedder in enumerate(self.inp_pos_embedders)
            ])
        elif isinstance(self.inp_pos_embedders, u.MxModule):
            inp_pos_embd = self.inp_pos_embedders(inp_idxs)
        else:
            raise ValueError(f"Unknown inp_pos_embedders type: {type(self.inp_pos_embedders)}")

        embd = self.prepend_begin_token(val_embd + inp_pos_embd)

        if self.tar_pos_embedders is not None:
            tar_idxs = inputs['ctx_tar_idxs']
            if isinstance(self.tar_pos_embedders, list):
                tar_pos_embd = tf.add_n([
                    embedder(tar_idxs[..., i])
                    for i, embedder in enumerate(self.tar_pos_embedders)
                ])
            elif isinstance(self.tar_pos_embedders, u.MxModule):
                tar_pos_embd = self.tar_pos_embedders(tar_idxs)
            else:
                raise ValueError(f"Unknown tar_pos_embedders type: {type(self.tar_pos_embedders)}")
            embd += tar_pos_embd
            embd /= tf.sqrt(tf.cast(3, u.dtype()))
        else:
            embd /= tf.sqrt(tf.cast(2, u.dtype()))

        return {
            'ctx_embd': embd,
        }


if __name__ == '__main__':
    import doctest
    doctest.testmod()
