#!/usr/bin/env python

import pickle
import os
import einops

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

from ml import predict, prediction_heads, utils

import config

try:
    run_name = os.environ["RUN_NAME"]
except KeyError:
    print()
    print("RUN_NAME not set. Must provide a name to run 'create_animation.py'.")
    exit(1)

model = keras.models.load_model(f"models/{run_name}", compile=False)
cfg = config.get()
predict_mean_fn = predict.create_predict_fn(cfg, prediction_heads.von_mises_dist, prediction_heads.von_mises_mean, model)
predict_sample_fn = predict.create_predict_fn(cfg, prediction_heads.von_mises_dist, prediction_heads.von_mises_sample, model)

from ml import data_bvh, data_tf

filenames, angles, n_frames = data_bvh.np_dataset_parallel_lists(cfg.force)
names = [os.path.basename(os.path.normpath(f)) for f in filenames]
all_angles = tf.concat(angles, axis=0)
circular_means = utils.circular_mean(all_angles, axis=0)[:cfg.n_hands, :cfg.n_dof]


i_example = 0
name = names[i_example]


## DATASET ANIMATIONS (subset)

example = tf.constant(angles[i_example])[1500:2600, :cfg.n_hands, :cfg.n_dof]
data_bvh.write_bvh_files(example, f"{run_name}.{name}", "anims/")

decimate = data_tf.make_decimate_fn(cfg)
example_decimated = decimate(example)
print("example_decimated", example_decimated.shape)
data_bvh.write_bvh_files(example_decimated, f"{run_name}.{name}.decimated", "anims/")


## CONDITIONAL ANIMATION

n_cond_frames = cfg.chunk_size-1
n_cond_frames_to_generate = len(example_decimated) - n_cond_frames
cond_angles = example_decimated[:n_cond_frames]
cond_frame_idxs = tf.tile(tf.range(n_cond_frames)[:, None, None], [1, cfg.n_hands, cfg.n_dof])
cond_hand_idxs = tf.tile(tf.range(cfg.n_hands)[None, :, None], [n_cond_frames, 1, cfg.n_dof])
cond_dof_idxs = tf.tile(tf.range(cfg.n_dof)[None, None, :], [n_cond_frames, cfg.n_hands, 1])
x = {
    "angles": tf.reshape(cond_angles, [1, -1]),
    "frame_idxs": tf.reshape(cond_frame_idxs, [1, -1]),
    "hand_idxs": tf.reshape(cond_hand_idxs, [1, -1]),
    "dof_idxs": tf.reshape(cond_dof_idxs, [1, -1]),
}
cond_means = predict_mean_fn(x, n_cond_frames_to_generate)
cond_samples = predict_sample_fn(x, n_cond_frames_to_generate)

i_batch = 0
means = einops.rearrange(cond_means[i_batch], '(seq hand dof) -> seq hand dof', hand=cfg.n_hands, dof=cfg.n_dof)
samples = einops.rearrange(cond_samples[i_batch], '(seq hand dof) -> seq hand dof', hand=cfg.n_hands, dof=cfg.n_dof)

if cfg.recluster:
    means = utils.unrecluster(means, circular_means)
    samples = utils.unrecluster(samples, circular_means)

data_bvh.write_bvh_files(means, f"{run_name}.{name}.means", "anims/")
data_bvh.write_bvh_files(samples, f"{run_name}.{name}.samples", "anims/")


## UNCONDITIONAL ANIMATIONS

test_batch_size = 2
x = {
    "angles": tf.zeros([test_batch_size, 0], dtype=tf.float32),
    "frame_idxs": tf.zeros([test_batch_size, 0], dtype=tf.int32),
    "hand_idxs": tf.zeros([test_batch_size, 0], dtype=tf.int32),
    "dof_idxs": tf.zeros([test_batch_size, 0], dtype=tf.int32),
}
y_pred_mean_batch = predict_mean_fn(x, 1100)
y_pred_sample_batch = predict_sample_fn(x, 1100)
for i_batch in range(y_pred_mean_batch.shape[0]):

    means = einops.rearrange(y_pred_mean_batch[i_batch], '(seq hand dof) -> seq hand dof', hand=cfg.n_hands, dof=cfg.n_dof)
    samples = einops.rearrange(y_pred_sample_batch[i_batch], '(seq hand dof) -> seq hand dof', hand=cfg.n_hands, dof=cfg.n_dof)

    if cfg.recluster:
        means = utils.unrecluster(means, circular_means)
        samples = utils.unrecluster(samples, circular_means)

    data_bvh.write_bvh_files(means, f"{run_name}.{i_batch}.means", "anims/")
    data_bvh.write_bvh_files(samples, f"{run_name}.{i_batch}.samples", "anims/")


## SHOW GUI AFTER GENERATING FILES

from ml import viz

for y_pred_mean, y_pred_sample in zip(y_pred_mean_batch, y_pred_sample_batch):
    viz.show_animations(cfg, [y_pred_mean, y_pred_sample])
viz.show_animations(cfg, [cond_means, cond_samples])
plt.show()
