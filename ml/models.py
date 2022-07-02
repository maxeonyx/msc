import tensorflow as tf
from tensorflow.keras import Model

class SequenceModelBase(Model):

    def __init__(self, cfg, is_recurrent=False, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.is_recurrent = is_recurrent

    def predict(self, x, n_frames):
        x = x.copy()

        batch_size = tf.shape(x["angles"])[0]

        if self.is_recurrent:
            output, state = self.recurrent_forward_pass(x)
        else:
            output = self(x, training=False)
        angles = self.prediction_head.mean(output)
        if self.prediction_head.is_distribution:
            angles_sample = self.prediction_head.sample(output)
        frame_idxs = x["frame_idxs"]
        hand_idxs = x["hand_idxs"]
        dof_idxs = x["dof_idxs"]
        
        start_frame = frame_idxs[..., -1] + 1
        for i_frame_offset in tf.range(0, n_frames):
            for i_hand in tf.range(self.cfg.n_hands):
                for i_dof in tf.range(self.cfg.n_dof):
                    # tile a constant value to the batch dimension and len=1 seq dim
                    tile_batch_seq = lambda x: tf.tile(x[None, None], [batch_size, 1])
                    
                    i_frame = start_frame + i_frame_offset
                    frame_idxs = tf.concat([frame_idxs, i_frame[..., None]], axis=1)

                    hand_idxs = tf.concat([hand_idxs, tile_batch_seq(i_hand)], axis=-1)
                    dof_idxs  = tf.concat([dof_idxs,  tile_batch_seq(i_dof)],  axis=-1)

                    input_dict = {
                        "angles": angles,
                        "frame_idxs": frame_idxs,
                        "hand_idxs": hand_idxs,
                        "dof_idxs": dof_idxs,
                    }
                    if self.is_recurrent:
                        output, state = self.single_forward_pass(input_dict, state)
                    else:
                        output = self(input_dict, training=False)[..., -1:, :] # model outputs a sequence, but we only need the new token

                    new_angles = self.prediction_head.mean(output)
                    if self.prediction_head.is_distribution:
                        new_angles_sample = self.prediction_head.sample(output)

                    angles = tf.concat([angles, new_angles], axis=-1)
                    if self.prediction_head.is_distribution:
                        angles_sample = tf.concat([angles_sample, new_angles_sample], axis=-1)
        
        # remove last output. because of the "begin" token, we only need to predict otherwise we have n+1 which doesn't evenly divide
        if self.prediction_head.is_distribution:
            return angles[..., :-1], angles_sample[..., :-1]
        return angles[..., :-1], None
