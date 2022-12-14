import pathlib
import tensorflow as tf
import typing
if typing.TYPE_CHECKING:
    import keras.api._v2.keras as keras
    from keras.api._v2.keras import Model, Input, layers
else:
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers
import einops as ein
from ml import utils, data_tf
from matplotlib import pyplot as plt
from box import Box as box
from math import pi, tau

import holoviews as hv
hv.extension('bokeh')

def create_predict_fn_v2(cfg, run_name, model, to_angle_fns):

    if type(model) is str:
        model = keras.models.load_model(model)

    figures = {}
    def build_figure(data, timestep, name="figure"):
        nonlocal figures
        n_imgs = len(data)
        names = list(data.keys())
        tracks = list(data.values())
        tracks = [track.numpy() if tf.is_tensor(track) else track for track in tracks]
        batch_size = tracks[0].shape[0]
        key_dims = [
            hv.Dimension(("batch", "Batch")),
            hv.Dimension(("time", "Timestep")),
        ]
        plots = {
            (timestep, i_batch): hv.Layout([
                hv.Raster(tracks[i_track][i_batch], label=names[i_track]).opts(cmap='twilight') for i_track in range(n_imgs)
            ], label=f"Batch {i_batch}", shared_axes="X")
            for i_batch in range(batch_size)
        }
        if name in figures:
            figures[name].update(plots)
        else:
            figures[name] = hv.HoloMap(plots, kdims=key_dims)
        h = figures[name].collate()
        pathlib.Path(f'_figures/{run_name}').mkdir(parents=True, exist_ok=True)
        hv.save(h, f'_figures/{run_name}/{name}.html')


    def showimgs(data, save, save_instead, t):
        n_imgs = len(data)
        names = list(data.keys())
        tracks = list(data.values())
        batch_size = tracks[0].shape[0]
        n_tracks_per_img = cfg.n_hands * cfg.n_joints_per_hand * cfg.n_dof_per_joint

        fig = plt.figure()
        fig.set_figheight(batch_size*n_imgs*1)
        subfigs = fig.subfigures(nrows=batch_size)
        subfigs.get
        for batch_i, sfig in enumerate(subfigs):
            sfig.set_in_layout(True)
            sfig.suptitle(f"Batch {batch_i + 1}")
            axes = sfig.subplots(nrows=n_imgs)
            for plot_i, (ax, title) in enumerate(zip(axes, names)):
                ax.set_aspect('equal')
                ax.set_title(title)
                ax.set_anchor('W')
                ax.imshow(tf.transpose(tf.reshape(tracks[plot_i][batch_i], [-1, n_tracks_per_img]))[:, :200], vmin=-pi, vmax=pi, interpolation='nearest', cmap='twilight')
        fig.tight_layout()
        if save or save_instead:
            if id is None:
                id = ""
            else:
                id = "_" + str(id)
            plt.savefig(f"_runs/{run_name}/fig{id}.png")
        if not save_instead:
            plt.show()

    @tf.function
    def predict_fn(seed_input, idxs, outp_var):
        n_stats = len(to_angle_fns)
        seed_len = seed_input.shape[1]
        # tile seed input across number of statistics we will generate (eg. mean() and sample())
        seed_input = ein.repeat(seed_input, 'b fh j d -> s b fh j d', s=n_stats)
        idxs = ein.repeat(idxs, "b fh i -> (s b) fh i", s=n_stats)
        outp_var[:, :, :seed_len].assign(seed_input)
        n = outp_var.shape[2]
        for i in tf.range(seed_len, n): # converted to tf.while_loop
            inp = ein.rearrange(outp_var[:, :, :i], "s b fh j d -> (s b) fh j d")
            inputs = {
                "input": inp,
                "input_idxs": idxs[:, :i],
                "target_idxs": idxs[:, i:i+1],
                "n_ahead": tf.constant(1, dtype=tf.int32),
            }
            output = model(inputs, training=False)
            output = output["output"]
            output = ein.rearrange(output, '(s b) (fh j d) params -> s b fh j d params', s=n_stats, j=cfg.n_joints_per_hand, d=cfg.n_dof_per_joint)
            for j, fn in enumerate(to_angle_fns.values()): # not converted, adds ops to graph
                vals = fn(output[j, :, -1, :, :, :])
                outp_var[j, :, i, :, :].assign(vals)
        return outp_var

    def make_seed_data(data, seed_len):
        return data["input"], data["input"][:, :seed_len], data["input_idxs"]

    def predict_wrapper(n_frames, seed_input, idxs):
        n_stats = len(to_angle_fns)
        batch_size = seed_input.shape[0]
        outp_var = tf.Variable(tf.zeros([n_stats, batch_size, n_frames * cfg.n_hands, cfg.n_joints_per_hand, cfg.n_dof_per_joint]))
        return predict_fn(seed_input, idxs, outp_var)

    def predict(data, n_frames, seed_len=8):
        target, seed_input, idxs = make_seed_data(data, seed_len)
        return predict_wrapper(n_frames, seed_input, idxs)

    def predict_and_show(data, n_frames, seed_len=8, timestep=0):
        target, seed_input, idxs = make_seed_data(data, seed_len)
        outp = predict_wrapper(n_frames, seed_input, idxs)
        build_figure(
            {
                "target": target,
                "seed": seed_input,
                **{ name: outp[i, ...] for i, (name) in enumerate(to_angle_fns.keys()) },
            },
            timestep=timestep,
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
