from box import Box

from mx.prelude import *
from mx.learned_mix_add import LearnedMixAdd

@export
class DecoderOnlyTransformer(u.MxModule):
    """
    GPT-style, decoder-only transformer.
    - Causal mask
    - Single sequence dimension
    - Layers consisting of
        - attention
        - feedforward
        - layer norm

    >>> model = DecoderOnlyTransformer(n_layers=2, n_heads=4, n_hidden=128, dropout=0.1, use_batchnorm=True, use_layernorm=True, use_learned_add=True)
    >>> model.build({'ctx_embd': tf.TensorShape([None, None, 512])})
    >>> model({ 'ctx_embd': tf.zeros([1, 10, 512]) }).shape
    TensorShape([1, 10, 512])
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        n_hidden: int,
        dropout: float = 0,
        use_batchnorm: bool = True,
        use_layernorm: bool = True,
        use_learned_add: bool = True,
        name="deconly",
        desc=None,
    ) -> None:
        super().__init__(
            name=name,
            desc=desc or f"DecoderOnlyTransformer({n_layers} layers, {n_heads} heads, {n_hidden} hidden units, and {dropout} dropout)",
        )

        self.n_layers = n_layers
        """Number of transformer layers"""
        self.n_heads = n_heads
        """Number of attention heads"""
        self.n_hidden = n_hidden
        """Hidden size of the feedforward layers"""
        self.dropout = dropout
        """Dropout rate. 0 = no dropout. Defaults to 0."""
        self.use_batchnorm = use_batchnorm
        """Whether to use batch normalization"""
        self.use_layernorm = use_layernorm
        """Whether to use layer normalization"""
        self.use_learned_add = use_learned_add
        """Whether to use learned addition instead of regular addition"""

    def build(self, input_shape):

        assert 'ctx_embd' in input_shape, f"Expected 'ctx_embd' in input_shape, got {input_shape}"

        n_embd = input_shape['ctx_embd'][-1]

        residual_blocks = []
        for i in range(self.n_layers):
            block = tf.Module(f"block{i}")
            residual_blocks.append(block)
            block.attn = layers.MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=n_embd,
                dropout=self.dropout,
                kernel_regularizer=None,
            )
            block.ffn = keras.Sequential()
            block.ffn.add(layers.Dropout(
                self.dropout,
            ))
            block.ffn.add(Dense(
                self.n_hidden,
                activation='gelu',
                kernel_regularizer=u.reg(),
            ))
            if self.use_batchnorm:
                block.ffn.add(layers.BatchNormalization())
            block.ffn.add(layers.Dropout(self.dropout))
            block.ffn.add(Dense(
                n_embd,
                kernel_regularizer=u.reg(),
            ))

            if self.use_learned_add:
                block.add_attn = LearnedMixAdd(
                    n_embd,
                )
                block.add_ffn = LearnedMixAdd(
                    n_embd,
                )
                block.add_layer = LearnedMixAdd(
                    n_embd,
                )
            if self.use_layernorm:
                block.layernorm1 = layers.LayerNormalization()
                block.layernorm2 = layers.LayerNormalization()
        self.residual_blocks = residual_blocks
        self.built = True

    def call(self, inputs):
        res = inputs["ctx_embd"]

        for b in self.residual_blocks:
            x = res

            if self.use_layernorm:
                x = b.layernorm1(x)

            x = b.attn(x, x)
            if self.use_learned_add:
                res = b.add_attn([res, x])
            else:
                res = res + x

            x = res

            if self.use_layernorm:
                x = b.layernorm2(x)
            x = b.ffn(x)

            if self.use_learned_add:
                res = b.add_ffn([res, x])
            else:
                res = res + x

        return res

@export
class Resnet(u.MxModule):
    """
    Resnet for sequence prediction. Limited to predicting based on only the previous 1 input.
    """

    def __init__(
        self,
        n_layers: int,
        n_hidden: int,
        dropout: float = 0,
        use_batchnorm: bool = True,
        use_learned_add: bool = True,
        name="resnet",
    ) -> None:
        super().__init__(
            name=name,
            desc=f"Resnet({n_layers} layers, {n_hidden} hidden units, and {dropout} dropout)",
        )

        self.n_layers = n_layers
        """Number of Resnet layers"""
        self.n_hidden = n_hidden
        """Hidden size of the Resnet layers"""
        self.dropout = dropout
        """Dropout rate. 0 = no dropout. Defaults to 0."""
        self.use_batchnorm = use_batchnorm
        """Whether to use batch normalization"""
        self.use_learned_add = use_learned_add
        """Whether to use learned addition instead of regular addition"""

    def build(self, input_shape) -> Model:

        assert 'ctx_embd' in input_shape, f"Expected 'ctx_embd' in input_shape, got {input_shape}"

        n_embd = input_shape['ctx_embd'][-1]

        residual_blocks = []
        for i in range(self.n_layers):
            block = Box()
            residual_blocks.append(block)
            block.ffn = keras.Sequential()
            block.ffn.add(layers.Dropout(
                self.dropout,
                name=f"{self.name}/l{i}/ffn/dropout1",
            ))
            block.ffn.add(Dense(
                self.n_hidden,
                activation='gelu',
                kernel_regularizer=u.reg(),
                name=f"{self.name}/l{i}/ffn/in",
            ))
            if self.use_batchnorm:
                block.ffn.add(layers.BatchNormalization(
                    name=f"{self.name}/l{i}/batchnorm",
                ))
            block.ffn.add(layers.Dropout(
                self.dropout,
                name=f"{self.name}/l{i}/ffn/dropout2",
            ))
            block.ffn.add(Dense(
                n_embd,
                kernel_regularizer=u.reg(),
                name=f"{self.name}/l{i}/ffn/out",
            ))

            if self.use_learned_add:
                block.add_layer = LearnedMixAdd(
                    n_embd,
                    name=f"{self.name}/l{i}/add_layer",
                )

        self.residual_blocks = residual_blocks

    def call(self, inputs):
        res = inputs["ctx_embd"]
        for l in self.residual_blocks:
            x = l.ffn(res)
            if self.use_learned_add:
                res = l.add_layer([res, x])
            else:
                res = res + x
        return res

@export
class LittleMLP(u.MxModule):
    """
    Debugging model that just does 2 dense layers.
    """

    def __init__(
        self,
        n_hidden: int,
        dropout: float = 0,
        name="mlp",
    ) -> None:
        super().__init__(
            name=name,
            desc=f"LittleMLP({n_hidden} hidden units, and {dropout} dropout)",
        )
        self.n_hidden = n_hidden
        """Hidden size of the dense layers"""
        self.dropout = dropout
        """Dropout rate. 0 = no dropout. Defaults to 0."""

    def build(self, input_shape):
        assert 'ctx_embd' in input_shape, f"Expected 'ctx_embd' in input_shape, got {input_shape}"
        n_embd = input_shape['ctx_embd'][-1]
        self.dropout1 = layers.Dropout(
            self.dropout,
        )
        self.dense_in = Dense(
            self.n_hidden,
            activation='gelu',
            kernel_regularizer=u.reg(),
            name=f"{self.name}/dense_in",
        )
        self.norm = keras.layers.BatchNormalization(
            name=f"{self.name}/norm",
        )
        self.dropout2 = layers.Dropout(
            self.dropout,
        )
        self.dense_out = Dense(
            n_embd,
            kernel_regularizer=u.reg(),
            name=f"{self.name}/dense_out",
        )

    def call(self, inputs):
        assert 'ctx_embd' in inputs, f"Expected 'ctx_embd' in inputs, got {inputs}"
        x = inputs["ctx_embd"]
        x = self.dropout1(x)
        x = self.dense_in(x)
        x = self.norm(x)
        x = self.dropout2(x)
        x = self.dense_out(x)
        return x


if __name__ == '__main__':
    import doctest
    doctest.testmod()
