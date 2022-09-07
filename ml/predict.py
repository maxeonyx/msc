from configparser import Interpolation
import tensorflow as tf
import tensorflow.keras as keras
import einops as ein
from ml import utils, data_tf
from matplotlib import pyplot as plt
from box import Box as box
from math import pi, tau

def create_predict_fn_v2(cfg, run_name, model, get_angle_fns):

    if type(model) is str:
        model = keras.models.load_model(model)

    def showimgs(data, save, save_instead, id):
        n_imgs = len(data)
        batch_size = data[0].shape[0]
        n_tracks_per_img = cfg.n_hands * cfg.n_joints_per_hand * cfg.n_dof_per_joint
        fig, axes = plt.subplots(n_imgs * batch_size, sharex=True, sharey=True, figsize=(10, 3*n_imgs*batch_size))
        for i in range(batch_size):
            for j in range(n_imgs):
                axes[i * n_imgs + j].set_anchor('W')
                axes[i * n_imgs + j].imshow(tf.transpose(tf.reshape(data[j][i], [-1, n_tracks_per_img]))[:, :200], vmin=-pi, vmax=pi, interpolation='nearest', cmap='twilight')
        fig.tight_layout()
        if save or save_instead:
            if id is None:
                id = ""
            else:
                id = "_" + str(id)
            plt.savefig(f"_runs/{run_name}/fig{id}.png")
        if not save_instead:
            plt.show()

    @tf.function(jit_compile=False)
    def predict_fn(seed_input, idxs, outp_var):
        n_stats = len(get_angle_fns)
        seed_len = seed_input.shape[1]
        # tile seed input across number of statistics we will generate (eg. mean() and sample())
        seed_input = ein.repeat(seed_input, 'b fh j d -> b s fh j d', s=n_stats)
        idxs = ein.repeat(idxs, "b fh i -> (b s) fh i", s=n_stats)
        outp_var[:, :, :seed_len].assign(seed_input)
        n = outp_var.shape[2]
        for i in tf.range(seed_len, n): # converted to tf.while_loop
            inp = ein.rearrange(outp_var[:, :, :i], "b stat fh j d -> (b stat) fh j d")
            inputs = {
                "input": inp,
                "input_idxs": idxs[:, :i],
                "target_idxs": idxs[:, i:i+1],
                "n_ahead": tf.constant(1, dtype=tf.int32),
            }
            output = model(inputs, training=False)
            output = output["output"]
            output = ein.rearrange(output, '(b s) (fh j d) params -> b s fh j d params', s=n_stats, j=cfg.n_joints_per_hand, d=cfg.n_dof_per_joint)
            for j in range(len(get_angle_fns)): # not converted, adds ops to graph
                vals = get_angle_fns[j](output[:, j, -1, :, :, :])
                outp_var[:, j, i, :, :].assign(vals)
        return outp_var

    def make_seed_data(data, seed_len):
        return data["input"], data["input"][:, :seed_len], data["input_idxs"]

    def predict_wrapper(n_frames, seed_input, idxs):
        n_stats = len(get_angle_fns)
        batch_size = seed_input.shape[0]
        outp_var = tf.Variable(tf.zeros([batch_size, n_stats, n_frames * cfg.n_hands, cfg.n_joints_per_hand, cfg.n_dof_per_joint]))
        return predict_fn(seed_input, idxs, outp_var)

    def predict(data, n_frames, seed_len=8):
        target, seed_input, idxs = make_seed_data(data, seed_len)
        return predict_wrapper(n_frames, seed_input, idxs)

    def predict_and_show(data, n_frames, seed_len=8, save=True, save_instead=True, id=None):
        target, seed_input, idxs = make_seed_data(data, seed_len)
        outp = predict_wrapper(n_frames, seed_input, idxs)
        showimgs(
            [
                target,
                seed_input,
                *[outp[:, i, ...] for i in range(len(get_angle_fns))],
            ],
            save=save,
            save_instead=save_instead,
            id=id,
        )

    return predict, predict_and_show

def create_predict_fn(cfg, dist_fn, get_angle_fn, model):
    """
    Create a predict function that does autoregressive sampling.
    """

    @tf.function
    def predict(x, n_frames):
        batch_size = x["angles"].shape[0]

        params = model(x, training=False)
        dist = dist_fn(params)
        angles = get_angle_fn(dist)

        frame_idxs = x["frame_idxs"]
        hand_idxs = x["hand_idxs"]
        dof_idxs = x["dof_idxs"]
        
        if frame_idxs.shape[1] == 0:
            start_frame = tf.zeros([batch_size, 1], dtype=tf.int32)
        else:
            start_frame = frame_idxs[:, -1:] + 1

        # use chunk size - 1 so that we stay in-distribution
        n_chunk_toks = (cfg.chunk_size - 1) * cfg.n_hands * cfg.n_dof

        # tile a constant value to the batch dimension and len=1 seq dim
        def tile_batch_seq(x):
            return tf.tile(x[None, None], [batch_size, 1])

        # produce F*H*D - 1 predictions
        # the minus one is because we don't need to predict the last frame
        # eg. if i=99, then we don't need to predict frame 100 (it's out of bounds)
        def cond(i, angles, frame_idxs, hand_idxs, dof_idxs):
            return tf.less(i, n_frames*cfg.n_hands*cfg.n_dof-1)
        
        def body(i, angles, frame_idxs, hand_idxs, dof_idxs):

            i_frame_offset = i // (cfg.n_hands * cfg.n_dof)
            i_frame = start_frame + i_frame_offset[None, None]
            i_hand = (i // cfg.n_dof) % cfg.n_hands
            i_dof = i % cfg.n_dof
            
            frame_idxs = tf.concat([frame_idxs, i_frame], axis=-1)

            hand_idxs = tf.concat([hand_idxs, tile_batch_seq(i_hand)], axis=-1)
            dof_idxs  = tf.concat([dof_idxs,  tile_batch_seq(i_dof)],  axis=-1)
            
            # use a fixed length context window to predict the next frame
            start_idx = tf.math.maximum(0, tf.shape(frame_idxs)[1]-n_chunk_toks)
            inputs = {
                "angles": angles[:, start_idx:],
                "frame_idxs": frame_idxs[:, start_idx:],
                "hand_idxs": hand_idxs[:, start_idx:],
                "dof_idxs": dof_idxs[:, start_idx:],
            }
            params = model(inputs, training=False) # model outputs a sequence, but we only need the new token
            dist = dist_fn(params[:, -1:, :])
            new_angles = get_angle_fn(dist)

            angles = tf.concat([angles, new_angles], axis=1)
            
            return i+1, angles, frame_idxs, hand_idxs, dof_idxs
        
        _i, angles, _frame_idxs, _hand_idxs, _dof_idxs = tf.while_loop(
            cond,
            body,
            loop_vars=[
                tf.constant(0),
                angles,
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
            ],
        )
        
        return angles
    
    return predict
