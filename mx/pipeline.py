from __future__ import annotations

import abc
from dataclasses import dataclass
import re
from typing import Callable, Literal, Type

from mx.prelude import *
from mx.utils import DSets
from mx.visualizer import StatefulVisualization, Visualization, Visualizer, VizCfg

@export
class MxDataset(abc.ABC):
    "Base class for datasets."

    def __init__(
        self,
        desc: str,
        name: str,
    ):
        self.desc = desc
        "Human-readable name"

        self.name = name
        "Unique, machine-readable identifier"

        ## set by dataset.configure(task) ##
        self.adapt_in: Callable[[DSets], DSets] = None
        """
        Provided by subclass impl of configure().
        Function to adapt a dataset from raw format to task-specific task-input format
        """

        self.adapt_out: Callable[[DSets], DSets] = None
        """
        Provided by subclass impl of configure().
        Function to adapt a dataset from task-specific task-out format (output of predict_fn) to raw format
        """


    @abc.abstractmethod
    def configure(self, task: Task):
        pass

    def adapt(self, dsets: DSets) -> DSets:
        """
        Adapt a dataset from raw format to task-specific task-input format
        """

        assert self.adapt_in is not None, "Must call dataset.configure(task) before dataset.adapt(ds)"

        dsets = dsets.map(self.adapt_in)
        dsets = dsets.cache()

        return dsets

    @abc.abstractmethod
    def load(self, force_cache_reload: bool) -> DSets:
        """
        Load the dataset and adapt for the configured task. Implementations
        should cache and snapshot the dataset to disk, unless this is a very
        small dataset, or there is some other reason not to. This can be done
        using the `snapshot_cache_split` function.
        """
        pass

    @abc.abstractmethod
    def get_visualizations(self, viz_batch_size, task_specific_predict_fn) -> dict[str, Visualization]:
        pass

    def _snapshot_and_split(self, d: tf.data.Dataset, buffer_size=None) -> DSets:
        """
        Split a dataset into train, val, and test sets, and snapshot the train set to disk.

        Args:
            d (tf.data.Dataset): The dataset to split.

            buffer_size (int, optional):
                The size of the buffer to use when shuffling the dataset.
                Defaults to None, which means the buffer size will be the same as the dataset size.

        Returns:
            dsets: A DSets instance containing the train, val, and test datasets.
        """
        n = d.cardinality().numpy()
        if n == 0:
            raise ValueError("Dataset is empty.")
        elif buffer_size is None:
            if n == -1:
                raise ValueError("Dataset cardinality is unknown - must specify a buffer size when calling `snapshot_cache_split`.")
            buffer_size = n
        d = d.snapshot(f"./_cache/tf/{self.name}", compression=None)
        d = d.shuffle(buffer_size=buffer_size, seed=self.split_seed)

        # default split is 80/10/10
        test_size = int(n * self.split[1])
        val_size = int(n * self.split[2])
        train_size = n - test_size - val_size
        dsets = DSets(
            test=d.take(test_size),

            # remove "extra" from train and val
            val=(
                d
                .map(lambda x: {**x, "extra": None })
                .skip(test_size)
                .take(val_size)
            ),
            train=(
                d
                .map(lambda x: {**x, "extra": None })
                .skip(test_size + val_size)
                .take(train_size)
            ),
        )
        assert dsets.train.cardinality().numpy() == train_size
        assert dsets.test.cardinality().numpy() == test_size
        assert dsets.val.cardinality().numpy() == val_size

        return dsets

@export
class Task(abc.ABC):

    @dataclass
    class DatasetSpecificConfig:
        n_input_dims: int
        "Dimensionality of input vectors (recieved by run from dataset)"

    @dataclass
    class ModelSpecificConfig:
        n_output_embd: int
        "Number of output embedding dimensions (recieved by final layer from model)"

    def __init__(
        self,
        name: str,
        identifier: str,
        does_batching: bool = False,
    ):
        self.name = name
        "Human-readable name"

        self.identifier = identifier
        "Unique, machine-readable identifier"

        self.does_batching = does_batching
        "Whether the task does its own batching"

        self.ds_config_type: Type[Task.DatasetSpecificConfig] = Task.DatasetSpecificConfig
        "Required dataset-specific config"

        self.model_config_type: Type[Task.ModelSpecificConfig] = Task.ModelSpecificConfig
        "Required model-specific config"

        ## Configured by self.recieve_dataset_config(cfg) ##
        self.ds_cfg = None
        ## Configured by self.recieve_model_config(cfg) ##
        self.model_cfg = None
        ## Configured by task.configure(embedding) ##
        self.adapt_in: Callable[[DSets], DSets] | None = None

    def recieve_dataset_config(self, cfg):
        assert isinstance(cfg, self.ds_config_type), f"Expected {self.ds_config_type}, got {type_name(cfg)}"
        self.ds_cfg = cfg

    def recieve_model_config(self, cfg):
        assert isinstance(cfg, self.model_config_type), f"Expected {self.model_config_type}, got {type_name(cfg)}"
        self.model_cfg = cfg

    @abc.abstractmethod
    def configure(self, cfg, embedding: Embedding):
        """
        Configure this task for outputting in the format required by the given embedding,
        and provide any task-specific config required by that embedding.
        """
        pass

    @abc.abstractmethod
    def process(self, dsets: DSets) -> DSets:
        pass

    def adapt(self, dsets: DSets) -> DSets:
        """
        Adapt a dataset from raw format to task-specific task-input format
        """

        assert self.adapt_in is not None, "Must call task.configure(embedding) before task.adapt(dsets)"

        dsets = dsets.map(self.adapt_in)

        return dsets

    @abc.abstractmethod
    def make_final_layer(self) -> tf.keras.layers.Layer:
        pass

    @abc.abstractmethod
    def make_loss_fn(self) -> Callable[..., tf.Tensor]:
        pass

    @abc.abstractmethod
    def make_predict_fn(self, model) -> Callable:
        pass

@export
class Embedding(abc.ABC):
    """
    Base class for embeddings.
    """

    @dataclass
    class TaskSpecificConfig:
        n_input_dims: int
        "Size of the input vector (recieved from the task)."

    def __init__(
        self,
        name: str,
        identifier: str,
        n_embd: int
    ):
        self.name = name
        "Human-readable name"

        self.identifier = identifier
        "Unique, machine-readable identifier"

        self.n_embd = n_embd
        "Number of embedding dimensions"

        self.task_config_type: Type[Embedding.TaskSpecificConfig] = Embedding.TaskSpecificConfig

        ## set by self.recieve_task_config(cfg) ##
        self.task_cfg: Embedding.TaskSpecificConfig = None

    def receive_task_config(self, cfg: TaskSpecificConfig):
        assert isinstance(cfg, self.task_config_type), f"Expected {self.task_config_type}, got {type_name(cfg)}"
        self.task_cfg = cfg

    @abc.abstractmethod
    def configure(self, model: MxModel):
        pass

    @abc.abstractmethod
    def make_embedder(self) -> Model:
        pass

@export
class MxModel(abc.ABC):

    @dataclass
    class EmbeddingSpecificConfig:
        n_embd: int
        "Number of embedding dimensions."

    def __init__(
        self,
        name: str,
        identifier: str,
    ):
        self.name = name
        "Human-readable name"

        self.identifier = identifier
        "Unique, machine-readable identifier"

        self.embd_cfg_type: Type[MxModel.EmbeddingSpecificConfig] = MxModel.EmbeddingSpecificConfig

        ## set by self.recieve_embedding_config(cfg) ##
        self.embd_cfg: MxModel.EmbeddingSpecificConfig = None

    def recieve_embd_config(self, cfg):
        assert isinstance(cfg, self.embd_cfg_type), f"Expected {self.embd_cfg_type}, got {type_name(cfg)}"
        self.embd_cfg = cfg

    @abc.abstractmethod
    def make_model(self):
        pass

    @abc.abstractmethod
    def configure(self, task: Task):
        pass

@export
@dataclass
class Pipeline:

    def __init__(self, batch_size: int,  n_steps: int, dataset: MxDataset, task: Task, embedding: Embedding, model: MxModel, name: str, desc: str, test_batch_size: int = None, viz_batch_size: int = 3) -> None:

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.viz_batch_size = viz_batch_size
        self.n_steps = n_steps

        # check identifier is machine-friendly
        assert re.match(r"^[a-z][a-z0-9_\-]+$", name), f"'{name}' is not a valid identifier, use only a-z, 0-9, _, -"

        self.name = name
        self.desc = desc

        # configure:
        #     raw data --> data generator
        dataset.configure(task)

        # configure:
        #     data generator --> input data
        task.configure(embedding)

        # configure:
        #     input data --> latents
        embedding.configure(model)

        # configure:
        #     output latents --> logits/outputs/params
        #     logits/outputs/params --> loss
        model.configure(task)

        self.dataset = dataset
        self.task = task
        self.embedding = embedding
        self.mx_model = model

        self._model: Model = None


    def new_model(self):
        "Create a new model instance."
        embedder = self.embedding.make_embedder()
        model = self.mx_model.make_model()
        final_layer = self.task.make_final_layer()

        self._model = Model(
            inputs=embedder.inputs,
            outputs=final_layer(model(embedder.outputs)),
            name=type(self.embedding).__name__ + '-' + type(self._model).__name__
        )

        return self._model

    def get_model(self) -> Model:
        "Create a new model, or return the existing one if it exists."
        if self._model is not None:
            return self._model

        return self.new_model()

    def get_loss_fn(self) -> Callable[..., tf.Tensor]:
        return self.task.make_loss_fn()

    def get_train_data(self, force_cache_reload: bool = False) -> tf.data.Dataset:

        dsets = self.dataset.load(force_cache_reload=force_cache_reload)
        dsets = self.dataset.adapt(dsets)
        dsets = self.task.process(dsets)
        dsets = self.task.adapt(dsets)

        d = dsets.train

        if not self.task.does_batching:
            d = d.batch(self.batch_size)

        d = (
            d
            .take(self.n_steps)
            .enumerate()
            .map(lambda i, x: (i, (x["inputs"], x["targets"])))
        )

        d = d.prefetch(tf.data.experimental.AUTOTUNE)

        return d

    def get_visualizer(self, output_dir, viz_cfgs: dict[str, VizCfg], viz_batch_size: int = None) -> Visualizer:

        viz_batch_size = viz_batch_size or self.viz_batch_size

        model = self.get_model()
        predict_fn = self.task.make_predict_fn(model)
        vizs = self.dataset.get_visualizations(viz_batch_size, predict_fn)
        return Visualizer(
            vizs,
            viz_cfgs,
            output_dir,
        )
