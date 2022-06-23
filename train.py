from cmath import e
import os
from trace import Trace

from matplotlib import pyplot as plt
import tensorboard
import tensorflow as tf
from tensorflow import keras

import config
from ml import data_bvh, data_tf, viz
from ml.embed import *
from ml.model_mlp import *
from ml.model_lstm import *
from ml.model_transformer import *
from ml.losses import *

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

prediction_head = MSE_angle()
embedder = AllIndicesConcatEmbedder(cfg)

# model = Dumb(cfg, embedder, prediction_head)
# model = Conv(cfg, embedder, prediction_head)
model = Transformer(cfg, embedder, prediction_head)
# model = RecurrentWrapper(cfg, ParallelLSTM(cfg), embedder, prediction_head)

# optimizer = keras.optimizers.Adam(learning_rate=WarmupLRSchedule(cfg.learning_rate, cfg.warmup_steps))
optimizer = keras.optimizers.Adam()
model.compile(loss=prediction_head.loss, optimizer=optimizer)

log_dir = f"./runs/{run_name}"
model.fit(d, steps_per_epoch=3000, epochs=8, callbacks=[
    viz.VizCallback(cfg, iter(d_test), log_dir + "/train"),
    keras.callbacks.TensorBoard(log_dir=log_dir),
])
print(f"Finished run '{run_name}'")
