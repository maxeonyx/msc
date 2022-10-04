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
    targets: dict[str, Einshape]
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
            train = self.train.map(fn),
            test = self.test.map(fn),
            val = self.val.map(fn),
        )
    
    def batch(self, batch_size, test_batch_size, shapes: DataPipelineShape) -> tuple[Self, DataPipelineShape]:
        dset = DSet(
            train = self.train.batch(batch_size),
            test = self.test.batch(test_batch_size),
            val = self.val.batch(test_batch_size),
        )

        train_shapes = DatasetShape(
            inputs={ k: shp.batch(batch_size) for k, shp in shapes.inputs.items() },
            targets={ k: shp.batch(batch_size) for k, shp in shapes.targets.items() },
            extra={ k: shp.batch(batch_size) for k, shp in shapes.extra.items() },
        )

        test_val_shapes = DatasetShape(
            inputs={ k: shp.batch(test_batch_size) for k, shp in shapes.inputs.items() },
            targets={ k: shp.batch(test_batch_size) for k, shp in shapes.targets.items() },
            extra={ k: shp.batch(test_batch_size) for k, shp in shapes.extra.items() },
        )

        shapes = DataPipelineShape(
            train=train_shapes,
            test=test_val_shapes,
            val=test_val_shapes,
        )

        return dset, shapes
