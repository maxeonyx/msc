import bpy
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print("Found GPUs:", physical_devices)
assert len(physical_devices) == 1, "Expected only 1 GPU. Set CUDA_VISIBLE_DEVICES or find another server."
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow_probability as tfp
import enlighten


import bpy_animate_addon

bpy_animate_addon.register()

from bpy import data as D
from bpy import context as C

C.scene.objects['leftHand'].select_set(False)
C.scene.objects['rightHand'].select_set(True)

bpy