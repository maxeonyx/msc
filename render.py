#!/usr/bin/env python

import pickle

import tensorflow as tf
from tensorflow import keras
from box import Box as b

run_name = "forward_snipe"

with open(f"models/{run_name}/custom_objects.pkl", "rb") as f:
    custom_objects = pickle.load(f)

with keras.util.custom_object_scope(custom_objects):
    model = keras.models.load_model(f"models/{run_name}")

cfg = b(
    batch_size = 3,
)

x = {
    "angles": tf.zeros([cfg.batch_size, 0], dtype=tf.float32),
    "frame_idxs": tf.zeros([cfg.batch_size, 0], dtype=tf.int32),
    "hand_idxs": tf.zeros([cfg.batch_size, 0], dtype=tf.int32),
    "dof_idxs": tf.zeros([cfg.batch_size, 0], dtype=tf.int32),
}

y_pred_mean_batch, y_pred_sample_batch = model.predict(x, None, n_frames=30)

from ml import viz

for y_pred_mean, y_pred_sample in zip(y_pred_mean_batch, y_pred_sample_batch):
    viz.show_animations(cfg, [y_pred_mean, y_pred_sample])
