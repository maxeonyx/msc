import tensorflow as tf
from keras import layers, Model, Sequential

from ml.model_transformer import FeedforwardWrapper

class Conv(FeedforwardWrapper):
    def __init__(self, cfg, embedder, prediction_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.embedder = embedder
        self.prediction_head = prediction_head
        self.conv_layer = layers.Conv1D(filters=cfg.conv.filters, kernel_size=cfg.conv.width_frames*cfg.n_hands*cfg.n_dof, activation="relu", padding='valid')

    def call(self, inputs):
        embd = self.embedder.embed_sequence_with_begin_sentinel(inputs, length=self.cfg.n_hands*self.cfg.n_dof)
        embd = self.conv_layer(embd)
        outputs = self.prediction_head.unembed(embd)
        return outputs

class Dumb(FeedforwardWrapper):
    def __init__(self, cfg, embedder, prediction_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.embedder = embedder
        self.prediction_head = prediction_head
        self.model = Sequential([
            layers.Dense(self.cfg.embd_dim),
            layers.ReLU(),
            layers.Dropout(self.cfg.dropout_rate),
            layers.Dense(self.cfg.embd_dim),
            layers.ReLU(),
            layers.Dropout(self.cfg.dropout_rate),
        ])

    def call(self, inputs):
        embd = self.embedder.embed_sequence_with_begin_sentinel(inputs, length=self.cfg.n_hands*self.cfg.n_dof)
        
        # offset by n_dof
        embd = embd[..., :-(self.cfg.n_hands*self.cfg.n_dof-1), :]
        
        embd = self.model(embd)
        
        outputs = self.prediction_head.unembed(embd)
        return outputs
