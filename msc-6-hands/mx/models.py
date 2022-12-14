from box import Box

from mx.prelude import *

from mx import layers as mx_layers
from mx.pipeline import MxModel, Task, Task_ModelConfig
from mx.utils import dtype

class MxLayer(layers.Layer):
    """Just adds required `name` and `desc` to Layer"""
    def __init__(self, name: str, desc: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.desc = desc

@export
class DecoderOnlyTransformer(MxModel):
    """
    GPT-style, decoder-only transformer.
    - Causal mask
    - Single sequence dimension
    - Layers consisting of
        - attention
        - feedforward
        - layer norm
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        n_hidden: int,
        dropout: float = 0,
        use_batchnorm: bool = True,
        use_layernorm: bool = True,
        name="deconly",
        desc=None,
    ) -> None:
        super().__init__(
            name=name,
            desc=desc or f"Decoder-only transformer with {n_layers} layers, {n_heads} heads, {n_hidden} hidden units, and dropout {dropout}",
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


    def configure(self, task: Task):

        assert self.embd_cfg is not None, "Must call embedding.configure(model) before calling model.configure(task)"

        # The dimensionality doesn't change -- easy.
        n_output_embd = self.embd_cfg.n_embd

        if task.model_config_type == Task_ModelConfig:
            task.recieve_model_config(task.model_config_type(
                n_output_embd,
            ))
        else:
            raise NotImplementedError(f"Model {type_name(self)} does not support Task {type_name(task)}. If using autoreload in IPython, try restarting the interpreter.")

    def compute_output_shape(self, input_shape):
        return input_shape["context/embd"]

    def make_model(self):
        n_embd = self.embd_cfg.n_embd
        residual_blocks = [
            Box(
                attn = layers.MultiHeadAttention(
                    num_heads=self.n_heads,
                    key_dim=n_embd,
                    dropout=self.dropout,
                    kernel_regularizer=None,
                    name=f"{self.name}/l{i}/attn",
                ),
                add_attn = mx_layers.learned_mix_add(
                    n_embd,
                    name=f"{self.name}/l{i}/add_attn",
                ),
                ffn = keras.Sequential([
                    Dense(
                        self.n_hidden,
                        activation='gelu',
                        kernel_regularizer=u.reg(),
                        name=f"{self.name}/l{i}/ffn/in",
                    ),
                    layers.Dropout(
                        self.dropout,
                        name=f"{self.name}/l{i}/ffn/dropout",
                    ),
                    Dense(
                        n_embd,
                        kernel_regularizer=u.reg(),
                        name=f"{self.name}/l{i}/ffn/out",
                    ),
                ]),
                add_ffn = mx_layers.learned_mix_add(
                    n_embd,
                    name=f"{self.name}/l{i}/add_ffn",
                ),
                layernorm = layers.LayerNormalization(
                    name=f"{self.name}/l{i}/layernorm",
                ),
                batchnorm = layers.BatchNormalization(
                    name=f"{self.name}/l{i}/batchnorm",
                ),
                add_layer = mx_layers.learned_mix_add(
                    n_embd,
                    name=f"{self.name}/l{i}/add_layer",
                ),
            )
            for i in range(self.n_layers)
        ]

        def call(inputs):
            embd = inputs["context/embd"]

            for b in residual_blocks:
                x = embd
                x = b.add_attn([x, b.attn(x, x, x, use_causal_mask=True)])

                x = b.add_ffn([x, b.ffn(x)])

                x = b.batchnorm(x)
                x = b.layernorm(x)

                embd = b.add_layer([embd, x])

            return embd

        inputs = u.input_dict(
            Input([None, n_embd], dtype=u.dtype(), name="context/embd"),
        )
        return Model(
            inputs=inputs,
            outputs=call(inputs),
            name=self.name,
        )

@export
class Resnet(MxModel):
    """
    Resnet for sequence prediction. Limited to predicting based on only the previous tokens.
    """

    def __init__(
        self,
        n_layers: int,
        n_hidden: int,
        dropout: float = 0,
        name="resnet",
    ) -> None:
        super().__init__(
            name=name,
            desc=f"Resnet with {n_layers} layers, {n_hidden} hidden units, and dropout {dropout}",
        )

        self.n_layers = n_layers
        """Number of Resnet layers"""
        self.n_hidden = n_hidden
        """Hidden size of the Resnet layers"""
        self.dropout = dropout
        """Dropout rate. 0 = no dropout. Defaults to 0."""

    def configure(self, task: Task):

        assert self.embd_cfg is not None, "Must call embedding.configure(model) before calling model.configure(task)"

        # The dimensionality doesn't change -- easy.
        n_output_embd = self.embd_cfg.n_embd

        if task.model_config_type == Task_ModelConfig:
            task.recieve_model_config(task.model_config_type(
                n_output_embd,
            ))
        else:
            raise NotImplementedError(f"Model {type_name(self)} does not support Task {type_name(task)}. If using autoreload in IPython, try restarting the interpreter.")

    def make_model(self) -> Model:

        assert self.embd_cfg is not None, "Must call task.configure(model) before calling make_model()"

        backbone_layers = [
            Box(
                dense_in = Dense(
                    self.n_hidden,
                    activation='gelu',
                    kernel_regularizer=u.reg(),
                    name=f"{self.name}/l{i}/dense_in",
                ),
                dense_out = Dense(
                    self.embd_cfg.n_embd,
                    kernel_regularizer=u.reg(),
                    name=f"{self.name}/l{i}/dense_out",
                ),
                dropout = layers.Dropout(
                    self.dropout,
                ),
                add = mx_layers.LearnedMixAdd(
                    name=f"{self.name}/l{i}/add",
                ),
                norm = keras.layers.BatchNormalization(
                    name=f"{self.name}/l{i}/norm",
                ),
            )
            for i in range(self.n_layers)
        ]

        def call(inputs):
            x = inputs["context/embd"]
            for l in backbone_layers:
                x = l.add([x, l.norm(l.dense_out(l.dropout(l.dense_in(x))))])
            return x

        inputs = u.input_dict(
            Input([None, self.embd_cfg.n_embd], name="context/embd"),
        )
        return Model(
            inputs=inputs,
            outputs=call(inputs),
            name=self.name,
        )

@export
class DebugMLP(MxModel):
    """
    Debugging model that just does 2 dense layers.
    """

    def __init__(
        self,
        n_hidden: int,
        dropout: float = 0,
        name="debug_mlp",
    ) -> None:
        super().__init__(
            name=name,
            desc="Debugging model that just does 2 dense layers.",
        )

        self.n_hidden = n_hidden
        """Hidden size of the dense layers"""
        self.dropout = dropout
        """Dropout rate. 0 = no dropout. Defaults to 0."""

    def configure(self, task: Task):

        assert self.embd_cfg is not None, "Must call embedding.configure(model) before calling model.configure(task)"

        # The dimensionality doesn't change -- easy.
        n_output_embd = self.embd_cfg.n_embd

        if task.model_config_type == Task_ModelConfig:
            task.recieve_model_config(task.model_config_type(
                n_output_embd,
            ))
        else:
            raise NotImplementedError(f"Model {type_name(self)} does not support Task {type_name(task)}. If using autoreload in IPython, try restarting the interpreter.")

    def make_model(self) -> Model:

        assert self.embd_cfg is not None, "Must call embedding.configure(model) before calling make_model()"

        layer1 = Dense(self.n_hidden, activation='relu', kernel_regularizer=u.reg())
        layer2 = Dense(self.embd_cfg.n_embd, activation='relu', kernel_regularizer=u.reg())

        inputs = u.input_dict(
            Input([None, self.embd_cfg.n_embd], name="context/embd"),
        )
        return Model(
            inputs=inputs,
            outputs=layer2(layer1(inputs["context/embd"])),
            name=self.name,
        )
