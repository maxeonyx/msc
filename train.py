from cmath import e
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input

import config
from ml import data_tf, predict, utils, viz, embedders, encoders, decoders

try:
    run_name = os.environ["RUN_NAME"]
except KeyError:
    import randomname
    run_name = randomname.get_name()
    os.environ["RUN_NAME"] = run_name

gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

cfg = config.get()
cfg.force = False
d, d_test = data_tf.tf_dataset(cfg)

embedder = embedders.add_embedder(cfg)
encoder = encoders.transformer(cfg | cfg.transformer)
loss_fn, decoder = decoders.von_mises(cfg)

inputs = {
    "angles": Input(shape=[None], dtype=tf.float32, name="angles"),
    "frame_idxs": Input(shape=[None], dtype=tf.int32, name="frame_idxs"),
    "hand_idxs": Input(shape=[None], dtype=tf.int32, name="hand_idxs"),
    "dof_idxs": Input(shape=[None], dtype=tf.int32, name="dof_idxs"),
}
embeddings = embedder(inputs)
latents = encoder(embeddings)
dist = decoder(latents)
model = keras.Model(inputs=list(inputs.values()), outputs=dist)

predictor = predict.create_predict_fn(cfg, model)

optimizer = keras.optimizers.SGD(momentum=0.9, learning_rate=utils.WarmupLRSchedule(cfg.learning_rate, cfg.warmup_steps))
# optimizer = keras.optimizers.Adam(learning_rate=WarmupLRSchedule(cfg.learning_rate, cfg.warmup_steps))
model.compile(loss=loss_fn, optimizer=optimizer, metrics=[utils.KerasLossWrapper(loss_fn)])

log_dir = f"./runs/{run_name}"
model.fit(d, steps_per_epoch=cfg.steps_per_epoch, epochs=cfg.steps//cfg.steps_per_epoch, callbacks=[
    viz.VizCallback(cfg, iter(d_test), predictor, log_dir + "/train"),
    keras.callbacks.TensorBoard(log_dir=log_dir),
])
print(f"Finished run '{run_name}'")

print(keras.utils.get_custom_objects())
model.summary(expand_nested=True)
model.save(f"models/{run_name}")
