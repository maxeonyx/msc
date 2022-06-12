from cmath import e

from matplotlib import pyplot as plt
import tensorflow as tf

import config
from ml import data_bvh, data_tf, model_lstm, viz

gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

cfg = config.get()
cfg.force = False
d, d_test = data_tf.tf_dataset(cfg)

el = next(iter(d_test))
print("element spec:", d.element_spec)
print("input el shape")
for k, v in el.items():
    print(k, v.shape)
model = model_lstm.ModelWrapper(cfg, architecture_model=model_lstm.LSTM(cfg))

model.compile(optimizer="adam", loss="mse")

model.fit(d, steps_per_epoch=300, epochs=1, callbacks=[
    viz.VizCallback(cfg, el)
])
