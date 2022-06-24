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

prediction_head = VonMises()
embedder = AllIndicesConcatEmbedder(cfg)

# model = Dumb(cfg, embedder, prediction_head)
# model = Conv(cfg, embedder, prediction_head)
model = KerasTransformer(cfg, embedder, prediction_head)
# model = RecurrentWrapper(cfg, ParallelLSTM(cfg), embedder, prediction_head)

optimizer = keras.optimizers.SGD(momentum=0.9, learning_rate=WarmupLRSchedule(cfg.learning_rate, cfg.warmup_steps))
# optimizer = keras.optimizers.Adam()
model.compile(loss=prediction_head.loss, optimizer=optimizer)

log_dir = f"./runs/{run_name}"
model.fit(d, steps_per_epoch=cfg.steps_per_epoch, epochs=cfg.steps//cfg.steps_per_epoch, callbacks=[
    viz.VizCallback(cfg, iter(d_test), log_dir + "/train"),
    keras.callbacks.TensorBoard(log_dir=log_dir),
])
print(f"Finished run '{run_name}'")
