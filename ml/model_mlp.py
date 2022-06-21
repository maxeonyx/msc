import tensorflow as tf
from keras import layers, Model, Sequential

class MLP(Model):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.needs_warmup = False

        self.layer_list = [layers.Dense(units, activation="relu") for units in cfg.mlp.layers]
    
    def __call__(self, embedded_sequence):
        
        for layer in self.layer_list:
            embedded_sequence = layer(embedded_sequence)

        return embedded_sequence, tf.constant(0)

    def predict_step(self, embedded_x, extra):

        return self(embedded_x)
