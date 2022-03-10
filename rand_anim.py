from venv import create
import tensorflow as tf

import create_dataset

data = tf.random.uniform(shape=(8000, 23), minval=-180, maxval=180, dtype=tf.float32).numpy()

create_dataset.write_bvh_file("./manipnet/Data/SimpleVisualizer/Assets/BVH/bottle1_body1/leftHand.bvh", data)
