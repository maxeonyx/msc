from platform import architecture
from numpy import zeros_like
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow import keras
from keras import layers, Model, Sequential


class LSTM(Model):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.needs_warmup = True
        self.lstm = layers.LSTM(cfg.embd_dim, return_sequences=True, return_state=True)
    
    def __call__(self, embedded_sequence):
        
        embedded_sequence, *states = self.lstm(embedded_sequence)

        return embedded_sequence, states
    
    def warmup(self, x):

        embedded_sequence, states = self(x)
        
        return embedded_sequence, states
    
    def predict_step(self, embedded_x, states):

        # remove sequence dimension for LstmCell
        embedded_x = embedded_x[..., 0, :]

        embedded_x, states = self.lstm.cell(embedded_x, states=states)
        
        # add sequence dimension back
        embedded_x = embedded_x[..., None, :]

        return embedded_x, states
        

class ModelWrapper(Model):
    def __init__(self, cfg, architecture_model, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        self.embd_angle = Sequential([
            layers.Dense(256),
            layers.ReLU(),
            layers.Dense(16),
        ])
        self.embd_hand_idxs = layers.Embedding(self.cfg.n_hands, 16)
        self.embd_frame_idxs = layers.Embedding(self.cfg.chunk_size, 16)
        self.embd_dof_idxs = layers.Embedding(self.cfg.n_dof, 16)
        self.embd_sentinel = layers.Embedding(1, 64)

        self.architecture_model = architecture_model

        self.unembed_layer = layers.Dense(1)

    def embed_inputs(self, inputs):

        # embed angles with a dense layer (add channel dim first)
        embd_angle = self.embd_angle(inputs["angles"][..., None])
        embd_frame_idxs = self.embd_frame_idxs(inputs["frame_idxs"])
        embd_hand_idxs = self.embd_hand_idxs(inputs["hand_idxs"])
        embd_dof_idxs = self.embd_dof_idxs(inputs["dof_idxs"])

        sentinel_vec = self.embd_sentinel(tf.zeros_like(inputs["hand_idxs"][..., -1:]))
        
        # concatenate the embeddings on the channel dim
        embd = tf.concat([embd_angle, embd_hand_idxs, embd_frame_idxs, embd_dof_idxs], axis=-1)
        
        # concatenate the BEGIN token on the seq dim
        embd = tf.concat([sentinel_vec, embd], axis=-2)

        return embd
    
    def unembed(self, embd):

        embd = self.unembed_layer(embd)
        
        # remove channel dim
        embd = embd[..., 0]

        return embd

    def call(self, inputs):

        embd = self.embed_inputs(inputs)

        embd, _extra = self.architecture_model(embd)

        unembed_angles = self.unembed(embd)

        return unembed_angles
    
    def predict(self, x, start_frame, n_frames):

        batch_size = tf.shape(x["angles"])[0]

        if self.architecture_model.needs_warmup:
            warmup_embd, extra = self.architecture_model(self.embed_inputs(x))
            new_angles = self.unembed(warmup_embd)
            angles = tf.concat([x["angles"], new_angles[..., -1:]], axis=-1)
        else:
            extra = None
            angles = x["angles"]

        for i_frame in tf.range(start_frame, start_frame + n_frames):
            for i_hand in tf.range(self.cfg.n_hands):
                for i_dof in tf.range(self.cfg.n_dof):
                    # tile a constant value to the batch dimension and len=1 seq dim
                    tile_batch_seq = lambda x: tf.tile(tf.constant(x)[None, None], [batch_size, 1])
                    embd = self.embed_inputs({
                        "angles": angles[..., -1:], # take last output as new input
                        "frame_idxs": tile_batch_seq(i_frame),
                        "hand_idxs": tile_batch_seq(i_hand),
                        "dof_idxs": tile_batch_seq(i_dof),
                    })
                    embd, extra = self.architecture_model.predict_step(embd, extra)

                    new_angles = self.unembed(embd)

                    angles = tf.concat([angles, new_angles], axis=1)

        return angles
