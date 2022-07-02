import tensorflow as tf
from keras import layers, Model

from ml.models import SequenceModelBase

class RNN(Model):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.needs_warmup = True
        self.layer = layers.RNN(cfg.embd_dim, dropout=cfg.dropout_rate, return_sequences=True, return_state=True)
    
    def call(self, embedded_sequence):
        
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
        self.layer = layers.LSTM(cfg.embd_dim, dropout=cfg.dropout_rate, return_sequences=True, return_state=True)
    
    def call(self, embedded_sequence):
        
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
        self.layer_list = [layers.LSTM(cfg.embd_dim, dropout=cfg.dropout_rate, return_sequences=True, return_state=True) for _ in range(cfg.multi_lstm.n_layers)]
    
    def call(self, embedded_sequence):
        
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
        self.layer_list = [layers.LSTM(cfg.embd_dim, dropout=cfg.dropout_rate, return_sequences=True, return_state=True) for _ in range(cfg.multi_lstm.n_layers)]
    
    def call(self, embedded_sequence):
        
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
        self.layer_list = [layers.LSTM(cfg.embd_dim, dropout=cfg.dropout_rate, return_sequences=True, return_state=True) for _ in range(cfg.multi_lstm.n_layers)]
    
    def call(self, embedded_sequence):
        
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
        self.layer_list = [layers.GRU(cfg.embd_dim, dropout=cfg.dropout_rate, return_sequences=True, return_state=True) for _ in range(cfg.multi_lstm.n_layers)]
    
    def call(self, embedded_sequence):
        
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

class RecurrentWrapper(SequenceModelBase):
    def __init__(self, cfg, encoder, embedder, prediction_head, **kwargs):
        super().__init__(cfg, is_recurrent=True, **kwargs)
        self.embedder = embedder
        self.prediction_head = prediction_head
        self.encoder = encoder

    def call(self, inputs):
        output, _state = self.recurrent_forward_pass(inputs)
        return output

    def recurrent_forward_pass(self, inputs):
        embd = self.embedder.embed_sequence_with_begin_sentinel(inputs)
        embd, state = self.encoder(embd)
        output = self.prediction_head.unembed(embd)
        return output, state

    def single_forward_pass(self, x, state):
        embd = self.embedder.embed_single({
            "angles": x["angles"][..., -1:], # take last output as new input
            "frame_idxs": x["frame_idxs"][..., -1:],
            "hand_idxs": x["hand_idxs"][..., -1:],
            "dof_idxs": x["dof_idxs"][..., -1:],
        })
        embd, state = self.encoder.predict_step(embd, state)
        output = self.prediction_head.unembed(embd)
        return output, state
    
    # def predict(self, x, y, n_frames):
    #     x = x.copy()

    #     batch_size = tf.shape(x["angles"])[0]
    #     warmup_embd = self.embedder.embed_sequence_with_begin_sentinel(x)
    #     warmup_embd, extra = self.architecture_model(warmup_embd)
    #     output = self.prediction_head.unembed(warmup_embd)
    #     angles = self.prediction_head.sample(output)
    #     frame_idxs = x["frame_idxs"]
    #     hand_idxs = x["hand_idxs"]
    #     dof_idxs = x["dof_idxs"]
        
    #     start_frame = frame_idxs[..., -1] + 1
    #     for i_frame in tf.range(start_frame, start_frame + n_frames - 1):
    #         for i_hand in tf.range(self.cfg.n_hands):
    #             for i_dof in tf.range(self.cfg.n_dof):
    #                 # tile a constant value to the batch dimension and len=1 seq dim
    #                 tile_batch_seq = lambda x: tf.tile(x[None, None], [batch_size, 1])
                    
    #                 if self.cfg.relative_frame_idxs and not self.cfg.target_is_sequence:
    #                     tile_batch = lambda x: tf.tile(x[None, :], [batch_size, 1])
    #                     frame_idxs = tile_batch(data_tf.frame_idxs_for(self.cfg, i_hand * self.cfg.n_dof + i_dof, tf.shape(angles)[-1]))
    #                 elif self.cfg.relative_frame_idxs:
    #                     raise NotImplementedError("Transformer does not yet support relative frame indices")
    #                 else:
    #                     frame_idxs = tf.concat([frame_idxs, tile_batch_seq(i_frame)], axis=1)
                    
    #                 hand_idxs = tf.concat([hand_idxs, tile_batch_seq(i_hand)], axis=-1)
    #                 dof_idxs  = tf.concat([dof_idxs,  tile_batch_seq(i_dof)],  axis=-1)
                    
    #                 new_angles = self.prediction_head.sample(output)

    #                 angles = tf.concat([angles, new_angles], axis=-1)

    #     return angles[..., :-1], None # remove last output because otherwise we have n+1 which doesn't evenly divide
