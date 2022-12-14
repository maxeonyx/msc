import os
import typing

import tensorflow as tf

print()
try:
    run_name = os.environ["RUN_NAME"]
except KeyError:
    import randomname
    run_name = randomname.get_name()
    os.environ["RUN_NAME"] = run_name
print(f"Starting run {run_name}.")
print()

print("Initializing tf ... ", end="", flush=True)
if typing.TYPE_CHECKING:
    from tensorflow.python import keras
    from tensorflow.python.keras import layers, Model, Input
else:
    from tensorflow import keras
    from tensorflow.keras import layers, Model, Input
if len(tf.config.list_physical_devices('GPU')) > 0:
    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)
tf.constant(1)
print("Done.")
print()

from ml import dream
import config

cfg = config.get()
cfg.force = False

dream.train(cfg, run_name)

exit()
