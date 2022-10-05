from dataclasses import dataclass
import abc
from typing import TypedDict
from typing_extensions import Self

import tensorflow as tf

from mx.utils import Einshape

@dataclass
class BaseDatasetConfig(abc.ABC):
    """
    Info common to all datasets defined in this repo.
    """
    pretty_name: str
    code_name: str

@dataclass
class DatasetShape:
    inputs: dict[str, Einshape]
    targets: Einshape
    extra: dict[str, Einshape]

@dataclass
class DataPipelineShape:
    train: DatasetShape
    test: DatasetShape
    val: DatasetShape

@dataclass
class DSet:
    train: tf.data.Dataset
    test: tf.data.Dataset
    val: tf.data.Dataset

    def destructure(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        return self.train, self.test, self.val

    def map(self, fn) -> Self:
        return DSet(
            train = self.train.map(fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
            test = self.test.map(fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
            val = self.val.map(fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
        )
    
    def batch(self, batch_size, test_batch_size, shapes: DataPipelineShape) -> tuple[Self, DataPipelineShape]:
        dset = DSet(
            train = self.train.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
            test = self.test.batch(test_batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
            val = self.val.batch(test_batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
        )

        train_shapes = DatasetShape(
            inputs={ k: shp.batch(batch_size) for k, shp in shapes.inputs.items() },
            targets=shapes.targets.batch(batch_size),
            extra={ k: shp.batch(batch_size) for k, shp in shapes.extra.items() },
        )

        test_val_shapes = DatasetShape(
            inputs={ k: shp.batch(test_batch_size) for k, shp in shapes.inputs.items() },
            targets=shapes.targets.batch(test_batch_size),
            extra={ k: shp.batch(test_batch_size) for k, shp in shapes.extra.items() },
        )

        shapes = DataPipelineShape(
            train=train_shapes,
            test=test_val_shapes,
            val=test_val_shapes,
        )

        return dset, shapes
