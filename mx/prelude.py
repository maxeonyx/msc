from math import pi, tau
import os
import sys
from pathlib import Path

import icecream as ic
import numpy as np
import einops as ein
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import keras_nlp
from tensorflow_probability import distributions as tfd
import typing
from typing import Callable, Literal, Union, Type, TypeVar
from typing_extensions import Self
from dataclasses import dataclass, KW_ONLY
if typing.TYPE_CHECKING:
    import keras.api._v2.keras as keras
    from keras.api._v2.keras import Model, Input, layers
    from keras.api._v2.keras.backend import is_keras_tensor
else:
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers
    from tensorflow.keras.backend import is_keras_tensor
    from tensorflow.types.experimental import TensorLike
    import tensorflow.types.experimental as tft

from mx.export import export, exporter

import mx.tf_types as tft

## utils.py imports this file, so can't use anything after this line
import mx.utils as u

from mx.utils import tf_print as tp, export, exporter, shape
