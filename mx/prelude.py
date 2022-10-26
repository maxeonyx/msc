from math import pi, tau, prod, sqrt
import os
import sys
from pathlib import Path
import typing
from typing import Callable, Literal, Union, Type, TypeVar
from typing_extensions import Self
from dataclasses import dataclass, KW_ONLY

from icecream import ic
from box import Box

import numpy as np
import einops as ein
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import keras_nlp
if typing.TYPE_CHECKING:
    import keras.api._v2.keras as keras
    import keras.api._v2.keras.activations
    import keras.api._v2.keras.models
    import keras.api._v2.keras.optimizers
    import keras.api._v2.keras.regularizers
    import keras.api._v2.keras.initializers
    import keras.api._v2.keras.constraints
    import keras.api._v2.keras.losses
    import keras.api._v2.keras.metrics
    import keras.api._v2.keras.callbacks
    import keras.api._v2.keras.backend
    from keras.api._v2.keras import Model, Input
    import keras.api._v2.keras.layers as layers
    from keras.api._v2.keras.layers import Embedding, Dense
    from keras.api._v2.keras.backend import is_keras_tensor
    from keras.api._v2.keras import mixed_precision as mixed_precision

    import tensorflow_probability.python.distributions as tfd
    import tensorflow_probability.python.layers as tfpl

    from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
else:
    from tensorflow import keras
    import tensorflow.keras.mixed_precision
    from tensorflow.keras import Input, Model, layers
    from tensorflow.keras.layers import Embedding, Dense
    from tensorflow.keras.backend import is_keras_tensor
    from tensorflow.types.experimental import TensorLike
    import tensorflow.types.experimental as tft
    from tensorflow.data import Dataset
    from tensorflow_probability import distributions as tfd
    from tensorflow_probability import layers as tfpl

from mx.export import export, exporter

import mx.tf_types as tft

## utils.py imports this file, so can't use anything after this line
import mx.utils as u

from mx.utils import dbg, tf_print as tp, export, exporter, shape, type_name
