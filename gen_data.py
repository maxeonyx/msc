
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
assert len(physical_devices) == 1, "Did not see the expected number of GPUs"
# to allow other tensorflow processes to use the gpu
# https://stackoverflow.com/a/60699372/7989988
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow_probability as tfp
import enlighten


def negloglik(targets, params):

    dist = tfp.layers.MixtureNormal(num_components = 3, event_shape=[])(params)

    loss = -dist.log_prob(targets)

    return loss

def von_mises_loss(targets, pred_params):

    loc = pred_params[:, :, 0]
    concentration = pred_params[:, :, 1]

    dist = tfp.distributions.VonMises(loc=loc, concentration=concentration)

    loss = -dist.log_prob(targets)

    return loss

dof = 23

def generate_data(wm, abs_model_path, conditioning_data, window_frames, new_frames):
    manager = enlighten.get_manager()
    
    model = tf.keras.models.load_model(
        abs_model_path,
        custom_objects={
            'negloglik': negloglik,
            'von_mises_loss': von_mises_loss
        },
    )

    conditioning_data = tf.cast(conditioning_data, tf.float32)

    prev_frames = min(window_frames, conditioning_data.shape[0])
    n_frames_to_predict = new_frames

    counter = manager.counter(total=n_frames_to_predict*dof)
    wm.progress_begin(0, n_frames_to_predict*dof)

    dof_idxs = tf.tile(tf.range(dof)[None, :], [prev_frames + 1, 1])
    frame_idxs = tf.tile(tf.range(prev_frames + 1)[:, None], [1, dof]) * dof
    idxs = dof_idxs + frame_idxs
    idxs = tf.reshape(idxs, [-1])

    def iteration(i, data):
        tar_idx = prev_frames*dof + i + 1
        inp_idxs = tf.range(i, prev_frames*dof + i)
        inp = data[-prev_frames*dof:]
        inp_len = tf.shape(inp_idxs)[0]
        tar_len = 1
        pred_params = model({
            "colors": inp[None, :],
            "inp_idxs": inp_idxs[None, :],
            "tar_idxs": tar_idx[None, None],
            "enc_mask": tf.zeros((inp_len, inp_len)),
            "dec_mask": tf.zeros((tar_len, inp_len)),
        })

        loc = pred_params[:, :, 0]
#        concentration = pred_params[:, :, 1]

#        dist = tfp.distributions.VonMises(loc=loc, concentration=concentration)
#        sample = dist.mean()
        sample = loc
        sample = tf.reshape(sample, [-1])
        data = tf.concat([data, sample], axis=0)
        return i+1, data

    @tf.function(
        input_signature=[tf.TensorSpec([None, dof])],
    )
    def do_batch(data):
        inp_data = tf.reshape(data[-prev_frames:, :], [prev_frames*dof])
        _i, out_data = tf.while_loop(
            cond=lambda i, data: i < dof,
            body=iteration,
            loop_vars=[
                tf.constant(0),
                inp_data
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None]),
            ],
        )
        return tf.reshape(out_data[-dof:], [1, dof])

#    test_data = tf.cast(create_dataset.load_one_bvh_file(filename, convert_deg_to_rad=True), tf.float32)
    generated_data = conditioning_data[:prev_frames, :]
    for i in range(n_frames_to_predict):
        result = do_batch(generated_data)
        generated_data = tf.concat([generated_data, result], axis=0)
        counter.update(incr=dof)
        wm.progress_update(i*dof)
    counter.close()

    return generated_data