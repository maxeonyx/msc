import keras_nlp

from mx.tf import *
from mx.layers import input_dict

from mx.pipeline import MxModel, Task

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

    def __init__(self, n_layers: int, n_heads: int, n_hidden: int, dropout: float = 0) -> None:
        super().__init__(
            name = "DecoderOnlyTransformer",
            identifier = "decoder_only_transformer",
        )

        self.n_layers = n_layers
        """Number of transformer layers"""
        self.n_heads = n_heads
        """Number of attention heads"""
        self.n_hidden = n_hidden
        """Hidden size of the feedforward layers"""
        self.dropout = dropout
        """Dropout rate. 0 = no dropout. Defaults to 0."""
    
    def configure(self, task: Task):

        assert self.embd_cfg is not None, "Must call embedding.configure(model) before calling model.configure(task)"

        # The dimensionality doesn't change -- easy.
        n_output_embd = self.embd_cfg.n_embd
        
        if task.model_config_type == Task.ModelSpecificConfig:
            task.recieve_model_config(task.model_config_type(
                n_output_embd,
            ))
        else:
            raise NotImplementedError(f"Model {type(self)} does not support task {type(task)}")

    def make_model(self) -> Model:

        assert self.embd_cfg is not None, "Must call task.configure(model) before calling make_model()"

        backbone = keras.Sequential([ 
            keras_nlp.layers.TransformerDecoder(
                intermediate_dim=self.n_hidden,
                num_heads=self.n_heads,
            )
            for _ in range(self.n_layers)
        ])

        def call(inputs):
            return backbone(inputs["embd"])

        inputs = input_dict(
            Input([None, self.embd_cfg.n_embd], name="embd"),
        )
        return Model(
            inputs=inputs,
            outputs=call(inputs),
            name=self.identifier,
        )
        
