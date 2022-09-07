#!/usr/bin/env python

import pickle
import os

import einops as ein
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

from ml import dream, predict, prediction_heads, utils, data_bvh, data_tf

import config

try:
    run_name = os.environ["RUN_NAME"]
except KeyError:
    print()
    print("RUN_NAME not set. Must provide a name to run 'create_animation.py'.")
    exit(1)

print("Loading model ... ", end="", flush=True)
model = keras.models.load_model(f"_models/{run_name}/model", compile=False)
model.load_weights(tf.train.latest_checkpoint(f"_models/{run_name}"))
cfg = config.get()
cfg = cfg | cfg.dream | cfg.dream.ds_real | cfg.dream.task_flat
loss_fn, stat_fns, prediction_head = prediction_heads.angular(cfg)
predict_fn, predict_and_show = predict.create_predict_fn_v2(cfg, run_name, model, stat_fns)
print("Done.")


print("Loading data ... ", end="", flush=True)
filenames, angles, n_frames = data_bvh.np_dataset_parallel_lists(cfg.force, columns=cfg.columns)
all_angles = tf.concat(angles, axis=0)
circular_means = utils.circular_mean(all_angles, axis=0) # all examples concatenated so the "frame" axis is the first axis

d_train, d_test, d_val = data_tf.tf_dataset(cfg, dream.flat_dataset, data_tf.bvh_data)
inp, tar = next(iter(d_test))
print("Done.")


## DATASET ANIMATIONS (subset)
i_from = inp["input_idxs"][:, 0, 0].numpy()
i_to = inp["input_idxs"][:, -1, 0].numpy()
length = i_to - i_from
assert np.all(length == length[0])
length = length[0]
input_angles = inp["input"]
batch_size = input_angles.shape[0]
input_angles = ein.rearrange(inp["input"], 'b (f h) j d -> b f h (j d)', h=cfg.n_hands, j=cfg.n_joints_per_hand, d=cfg.n_dof_per_joint)
orig_angles = inp["orig_angles"]

examples = []
def write_files(data, f_ext):
    global examples
    if cfg.recluster:
        data = utils.unrecluster(data, circular_means, n_batch_dims=1)
    for i in range(batch_size):
        name = inp["filename"][i].numpy().decode("utf-8") 
        if len(examples) <= i:
            examples.append({ "name": name, "files": [] })
        f_ext_text = f_ext[i] if type(f_ext) == list else f_ext
        f_name = f"{run_name}.{name}.{i}{f_ext_text}"
        data_bvh.write_bvh_files(data[i], f_name, column_map=data_bvh.COL_ALL_JOINTS, output_dir="_anims/")
        examples[i]["files"].append(f_name  )

write_files(orig_angles, ".original")
write_files(input_angles, ".target")

## CONDITIONAL ANIMATION

n_seed_frames = 30
n_cond_frames_to_generate = length - n_seed_frames
print()
print("INFO: n_seed_frames:", n_seed_frames)
print("INFO: n_frames:", n_cond_frames_to_generate)
print("INFO: batch_size:", batch_size)
print("INFO: generating ... ", end="", flush=True)
seqs = predict_fn(inp, tf.constant(n_cond_frames_to_generate), tf.constant(n_seed_frames))
print("Done.")
print()

seqs = ein.rearrange(seqs, 'b s (f h) j d -> s b f h (j d)', h=cfg.n_hands, j=cfg.n_joints_per_hand, d=cfg.n_dof_per_joint)
means = seqs[0]
write_files(means[:, n_seed_frames:], ".predicted")
if len(seqs) > 1:
    samples = seqs[1]
    write_files(samples[:, n_seed_frames:], ".predicted.sampled")

import json
with open(f"_anims/{run_name}.json", "w") as f:
    json.dump({
        "n_seed_frames": n_seed_frames,
        "n_hands": cfg.n_hands,
        "decimated": cfg.decimate,
        "reclustered": cfg.recluster,
        "examples": examples,
    }, f, indent=1)

print("Done. Wrote animations to '_anims/'.")
print()
