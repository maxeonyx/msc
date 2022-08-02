from cmath import e
import os

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers, Model, Input

import config
from ml import data_tf, predict, utils, viz, embedders, encoders, decoders

try:
    run_name = os.environ["RUN_NAME"]
except KeyError:
    import randomname
    run_name = randomname.get_name()
    os.environ["RUN_NAME"] = run_name

if len(tf.config.list_physical_devices('GPU')) > 0:
    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)

cfg = config.get()
cfg.force = False
d, d_test = data_tf.tf_dataset(cfg)

def create_model(cfg, encoder_type):

    cfg = cfg | cfg[encoder_type]
    # create model
    inputs = {
        "vectors": Input(shape=[None], dtype=tf.float32, name="vectors"),
        "angles": Input(shape=[None], dtype=tf.float32, name="angles"),
        "frame_idxs": Input(shape=[None], dtype=tf.int32, name="frame_idxs"),
        "hand_idxs": Input(shape=[None], dtype=tf.int32, name="hand_idxs"),
        "dof_idxs": Input(shape=[None], dtype=tf.int32, name="dof_idxs"),
    }
    embedder = embedders.add_embedder(cfg)

    if encoder_type == "transformer":
        encoder = encoders.transformer(cfg | cfg.transformer)
    elif encoder_type == "mlp":
        encoder = encoders.mlp(cfg | cfg.mlp)
    elif encoder_type == "conv":
        encoder = encoders.conv(cfg | cfg.conv)

    loss_fn, dist_fn, mean_fn, sample_fn, decoder = decoders.von_mises(cfg)
    embeddings = embedder(inputs)
    latents = encoder(embeddings)
    params = decoder(latents)
    
    model = keras.Model(inputs=list(inputs.values()), outputs=params)
    predict_mean_fn = predict.create_predict_fn(cfg, dist_fn, mean_fn, model)
    predict_sample_fn = predict.create_predict_fn(cfg, dist_fn, sample_fn, model)

    return loss_fn, predict_mean_fn, predict_sample_fn, model

loss_fn, predict_mean_fn, predict_sample_fn, model = create_model(cfg, "transformer")

model.summary(expand_nested=True)

if cfg.optimizer == "warmup_sgd":
    optcfg = cfg | cfg.warmup_sgd
    optimizer = keras.optimizers.SGD(momentum=optcfg.momentum, clipnorm=optcfg.clip_norm, learning_rate=utils.WarmupLRSchedule(optcfg.lr, optcfg.warmup_steps))
# optimizer = keras.optimizers.Adam(learning_rate=WarmupLRSchedule(cfg.learning_rate, cfg.warmup_steps))
model.compile(loss=loss_fn, optimizer=optimizer, metrics=[utils.KerasLossWrapper(loss_fn)])

log_dir = f"./runs/{run_name}"
try:
    model.fit(d, steps_per_epoch=cfg.steps_per_epoch, epochs=cfg.steps//cfg.steps_per_epoch, callbacks=[
        viz.VizCallback(cfg, iter(d_test), [predict_mean_fn, predict_sample_fn], log_dir + "/train"),
        keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100),])
    print()
    print(f"Finished training '{run_name}'")
    print()
except KeyboardInterrupt:
    print()
    print(f"Exited training early for run '{run_name}'")
    print()

model.save(f"models/{run_name}")
