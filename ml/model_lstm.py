import math
import tensorflow as tf
from keras import layers, Model

from . import data_tf

class RNN(Model):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.needs_warmup = True
        self.layer = layers.RNN(cfg.embd_dim, dropout=cfg.dropout, return_sequences=True, return_state=True)
    
    def __call__(self, embedded_sequence):
        
        embedded_sequence, *states = self.layer(embedded_sequence)

        return embedded_sequence, states
    
    def predict_step(self, embedded_x, states):

        # remove sequence dimension for LstmCell
        embedded_x = embedded_x[..., 0, :]

        embedded_x, states = self.layer.cell(embedded_x, states=states)
        
        # add sequence dimension back
        embedded_x = embedded_x[..., None, :]

        return embedded_x, states

class LSTM(Model):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.needs_warmup = True
        self.layer = layers.LSTM(cfg.embd_dim, dropout=cfg.dropout, return_sequences=True, return_state=True)
    
    def __call__(self, embedded_sequence):
        
        embedded_sequence, *states = self.layer(embedded_sequence)

        return embedded_sequence, states
    
    def predict_step(self, embedded_x, states):

        # remove sequence dimension for LstmCell
        embedded_x = embedded_x[..., 0, :]

        embedded_x, states = self.layer.cell(embedded_x, states=states)
        
        # add sequence dimension back
        embedded_x = embedded_x[..., None, :]

        return embedded_x, states


class StackedLSTM(Model):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.needs_warmup = True
        self.layer_list = [layers.LSTM(cfg.embd_dim, dropout=cfg.dropout, return_sequences=True, return_state=True) for _ in range(cfg.multi_lstm.n_layers)]
    
    def __call__(self, embedded_sequence):
        
        embd = embedded_sequence
        states = []
        for lstm in self.layer_list:
            embd, *state = lstm(embd)
            states.append(state)
        
        return embd, states
    
    def predict_step(self, inp_embd, inp_states):

        # remove sequence dimension for LstmCell
        inp_embd = inp_embd[..., 0, :]

        embd = inp_embd
        out_states = []
        for lstm, state in zip(self.layer_list, inp_states):
            embd, state = lstm.cell(embd, states=state)
            out_states.append(state)
        out_embd = embd

        # add sequence dimension back
        out_embd = out_embd[..., None, :]

        return out_embd, out_states


class ResidualLSTM(Model):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.needs_warmup = True
        self.layer_list = [layers.LSTM(cfg.embd_dim, dropout=cfg.dropout, return_sequences=True, return_state=True) for _ in range(cfg.multi_lstm.n_layers)]
    
    def __call__(self, embedded_sequence):
        
        embd = embedded_sequence
        states = []
        for lstm in self.layer_list:
            x, *state = lstm(embd)
            embd += x
            states.append(state)
        
        return embd, states
    
    def predict_step(self, inp_embd, inp_states):

        # remove sequence dimension for LstmCell
        inp_embd = inp_embd[..., 0, :]

        embd = inp_embd
        out_states = []
        for lstm, state in zip(self.layer_list, inp_states):
            x, state = lstm.cell(embd, states=state)
            embd += x
            out_states.append(state)
        out_embd = embd

        # add sequence dimension back
        out_embd = out_embd[..., None, :]

        return out_embd, out_states

class ParallelLSTM(Model):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.needs_warmup = True
        self.layer_list = [layers.LSTM(cfg.embd_dim, dropout=cfg.dropout, return_sequences=True, return_state=True) for _ in range(cfg.multi_lstm.n_layers)]
    
    def __call__(self, embedded_sequence):
        
        out_embds, out_states = [], []
        for lstm in self.layer_list:
            embd, *state = lstm(embedded_sequence)
            out_embds.append(embd)
            out_states.append(state)
        
        out_embd = tf.add_n(out_embds)
        
        return out_embd, out_states
    
    def predict_step(self, inp_embd, inp_states):

        # remove sequence dimension for LstmCell
        inp_embd = inp_embd[..., 0, :]

        out_embds, out_states = [], []
        for lstm, state in zip(self.layer_list, inp_states):
            embd, state = lstm.cell(inp_embd, states=state)
            out_embds.append(embd)
            out_states.append(state)
        
        out_embd = tf.add_n(out_embds)

        # add sequence dimension back
        out_embd = out_embd[..., None, :]

        return out_embd, out_states


class ResidualGRU(Model):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.needs_warmup = True
        self.layer_list = [layers.GRU(cfg.embd_dim, dropout=cfg.dropout, return_sequences=True, return_state=True) for _ in range(cfg.multi_lstm.n_layers)]
    
    def __call__(self, embedded_sequence):
        
        embd = embedded_sequence
        states = []
        for lstm in self.layer_list:
            x, *state = lstm(embd)
            embd += x
            states.append(state)
        
        return embd, states
    
    def predict_step(self, inp_embd, inp_states):

        # remove sequence dimension for LstmCell
        inp_embd = inp_embd[..., 0, :]

        embd = inp_embd
        out_states = []
        for layer, state in zip(self.layer_list, inp_states):
            x, state = layer.cell(embd, states=state)
            embd += x
            out_states.append(state)
        out_embd = embd

        # add sequence dimension back
        out_embd = out_embd[..., None, :]

        return out_embd, out_states

class ModelWrapper(Model):
    def __init__(self, cfg, architecture_model, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        self.architecture_model = architecture_model

    def call(self, inputs):

        embd = self.embed_sequence_with_begin_sentinel(inputs)

        embd, _extra = self.architecture_model(embd)

        unembed_angles = self.unembed(embd)
        if not self.cfg.target_is_sequence:
            unembed_angles = unembed_angles[..., -1] # last prediction only

        return unembed_angles
    
    def predict(self, x, start_frame, n_frames):

        batch_size = tf.shape(x["angles"])[0]

        warmup_embd, extra = self.architecture_model(self.embed_sequence_with_begin_sentinel(x))
        angles = self.unembed(warmup_embd)
        angles = [angles]
        for _frame in tf.range(start_frame, start_frame + n_frames):
            for i_hand in tf.range(self.cfg.n_hands):
                for i_dof in tf.range(self.cfg.n_dof):
                    # tile a constant value to the batch dimension and len=1 seq dim
                    tile_batch_seq = lambda x: tf.tile(x[None, None], [batch_size, 1])
                    frame_idx = data_tf.frame_idxs_for(self.cfg, i_hand * self.cfg.n_dof + i_dof, 1)[0]

                    embd = self.embed_single({
                        "angles": angles[..., -1:], # take last output as new input
                        "frame_idxs": tile_batch_seq(frame_idx),
                        "hand_idxs": tile_batch_seq(i_hand),
                        "dof_idxs": tile_batch_seq(i_dof),
                    })
                    embd, extra = self.architecture_model.predict_step(embd, extra)

                    new_angles = self.unembed(embd)
                    
                    angles.append(new_angles)
        
        angles = tf.concat(angles, axis=-1)

        return angles[..., :-1] # remove last output because otherwise we have n+1 which doesn't evenly divide
