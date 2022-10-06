from dataclasses import KW_ONLY, dataclass
import abc
from typing import Callable, Type
from typing_extensions import Self

import tensorflow as tf
from mx import tasks

from . import bvh

from mx.utils import Einshape
from mx.tasks import Task

@dataclass
class DSets:
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
    
    def batch(self, batch_size, test_batch_size) -> Self:
        dset = DSet(
            train = self.train.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
            test = self.test.batch(test_batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
            val = self.val.batch(test_batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False),
        )

        return dset

    def cache(self) -> Self:
        return DSet(
            train = self.train.cache(),
            test = self.test.cache(),
            val = self.val.cache(),
        )

DsToTaskAdaptor = Callable[[tf.data.Dataset], DSets]

def snapshot_cache_split(d: tf.data.Dataset, identifier, split, seed, buffer_size=None) -> DSets:
    """Split a dataset into train, val, and test sets, and snapshot the train set to a cache.

    Args:
        d (tf.data.Dataset): The dataset to split.
        identifier (str): The identifier for the dataset.
        split (tuple, optional): The train/test/val split to use.
        seed (int, optional): The seed to use for shuffling.

    Returns:
        dict: A dictionary containing the train, val, and test datasets.
    """
    n = d.cardinality().numpy()
    if n == 0:
        raise ValueError("Dataset is empty.")
    elif buffer_size is None:
        if n == -1:
            raise ValueError("Dataset cardinality is unknown - must specify a buffer size when calling `snapshot_cache_split`.")
        buffer_size = n
    d = d.snapshot(f"./_cache/tf/{identifier}", compression=None)
    d = d.shuffle(buffer_size=buffer_size, seed=seed)

    # default split is 80/10/10
    test_size = int(n * split[1])
    val_size = int(n * split[2])
    train_size = n - test_size - val_size
    dsets = DSets(
        test=d.take(test_size),
        val=d.skip(test_size).take(val_size),
        train=d.skip(test_size + val_size).take(train_size),
    )
    assert dsets.train.cardinality().numpy() == train_size
    assert dsets.test.cardinality().numpy() == test_size
    assert dsets.val.cardinality().numpy() == val_size

    dsets = dsets.cache()

    return dsets


@dataclass
class Dataset(abc.ABC):

    name: str
    "Human-readable name"

    identifier: str
    "Unique, machine-readable identifier"

    _: KW_ONLY

    split = (0.8, 0.1, 0.1)
    "Ratios of train/test/val split"

    split_seed = 1234
    "Change this to split different data into train/test/val sets"

    @property
    @abc.abstractmethod
    def implementations(self) -> dict[Type[Task], DsToTaskAdaptor]:
        pass

    @abc.abstractmethod
    def load(self) -> tf.data.Dataset:
        """
        Load the dataset. Implementations should cache and snapshot the dataset to disk,
        unless this is a very small dataset, or there is some other reason not to. This
        can be done using the `snapshot_cache_split` function.
        """
        pass
    
    @abc.abstractmethod
    def visualize(self, predictions):
        pass

    @abc.abstractmethod
    def to_original_format(self, predicitons):
        pass
    
    def is_compatible_with(self, task: Task) -> bool:
        return type(task) in self.implementations

    def load_and_adapt_for(self, task: Task) -> DSets:
        if type(task) not in self.implementations:
            raise NotImplementedError(f"Task {task.name} not implemented for dataset {self.name}")
        else:
            adaptor = self.implementations[type(task)]

        orig_data = self.load()

        task_input_data = adaptor(orig_data)

        return task_input_data

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


def init_data_pipeline(
    data_cfg: BaseDatasetConfig,
    task_cfg: tasks.TaskCfg,
    train_cfg: tasks.TrainingCfg,
    force_cache_reload: bool = False,
) -> tuple[DSet, DataPipelineShape]:
    # create task
    if isinstance(data_cfg, bvh.BvhAllColumns):

        if isinstance(task_cfg, tasks.NextVectorPrediction):
            dset, shapes = bvh.vector_ntp(data_cfg, task_cfg, force_cache_reload=force_cache_reload)

    if not task_cfg.already_batched():
        dset, shapes = dset.batch(train_cfg.batch_size, train_cfg.test_batch_size, shapes)
    
    dset.train = (
        dset.train
        .take(8000)
        .enumerate()
        .map(lambda i, x: (i, (x["inputs"], x["targets"])))
    )

    return dset, shapes
