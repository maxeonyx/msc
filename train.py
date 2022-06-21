from cmath import e

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

import config
from ml import data_bvh, data_tf, viz
from ml.embed import *
from ml.model_mlp import *
from ml.model_lstm import *
from ml.model_transformer import *

gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

cfg = config.get()
cfg.force = False
d, d_test = data_tf.tf_dataset(cfg)

embedder = Embedder(cfg)

model = Transformer(cfg, embedder)

optimizer = keras.optimizers.Adam(learning_rate=WarmupLRSchedule(cfg.learning_rate, cfg.warmup_steps))
model.compile(optimizer=optimizer, loss="mse")

model.fit(d, steps_per_epoch=3000, epochs=10, callbacks=[
    viz.VizCallback(cfg, iter(d_test))
])
