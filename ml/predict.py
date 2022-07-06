import tensorflow as tf

def create_predict_fn(cfg, dist_fn, model):
    """
    Create a predict function that does autoregressive sampling.
    """

    @tf.function
    def predict(x, n_frames):
        batch_size = x["angles"].shape[0]

        params = model(x, training=False)
        dist = dist_fn(params)
        angles = dist.mean()
        angles_sample = dist.sample()

        frame_idxs = x["frame_idxs"]
        hand_idxs = x["hand_idxs"]
        dof_idxs = x["dof_idxs"]
        
        if frame_idxs.shape[1] == 0:
            start_frame = tf.zeros([batch_size, 1], dtype=tf.int32)
        else:
            start_frame = frame_idxs[:, -1:] + 1

        # tile a constant value to the batch dimension and len=1 seq dim
        def tile_batch_seq(x):
            return tf.tile(x[None, None], [batch_size, 1])

        # produce F*H*D - 1 predictions
        # the minus one is because we don't need to predict the last frame
        # eg. if i=99, then we don't need to predict frame 100 (it's out of bounds)
        def cond(i, angles, angles_sample, frame_idxs, hand_idxs, dof_idxs):
            return tf.less(i, n_frames*cfg.n_hands*cfg.n_dof-1)
        
        def body(i, angles, angles_sample, frame_idxs, hand_idxs, dof_idxs):

            i_frame_offset = i // (cfg.n_hands * cfg.n_dof)
            i_frame = start_frame + i_frame_offset[None, None]
            i_hand = (i // cfg.n_dof) % cfg.n_hands
            i_dof = i_frame_offset % cfg.n_dof
            
            frame_idxs = tf.concat([frame_idxs, i_frame], axis=-1)

            hand_idxs = tf.concat([hand_idxs, tile_batch_seq(i_hand)], axis=-1)
            dof_idxs  = tf.concat([dof_idxs,  tile_batch_seq(i_dof)],  axis=-1)

            inputs = {
                "angles": angles,
                "frame_idxs": frame_idxs,
                "hand_idxs": hand_idxs,
                "dof_idxs": dof_idxs,
            }
            params = model(inputs, training=False) # model outputs a sequence, but we only need the new token
            dist = dist_fn(params[:, -1:, :])
            new_angles = dist.mean()
            new_angles_sample = dist.sample()

            angles = tf.concat([angles, new_angles], axis=-1)
            angles_sample = tf.concat([angles_sample, new_angles_sample], axis=-1)
            
            return i+1, angles, angles_sample, frame_idxs, hand_idxs, dof_idxs
        
        _i, angles, angles_sample, _frame_idxs, _hand_idxs, _dof_idxs = tf.while_loop(
            cond,
            body,
            loop_vars=[
                tf.constant(0),
                angles,
                angles_sample,
                frame_idxs,
                hand_idxs,
                dof_idxs,
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
        )
        
        return angles, angles_sample
    
    return predict
