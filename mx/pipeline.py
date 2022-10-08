from __future__ import annotations

import abc
from dataclasses import dataclass
import re
from typing import Callable, Type

from mx.tf import *
from mx.utils import DSets

@export
class MxDataset(abc.ABC):
    "Base class for datasets."

    def __init__(
        self,
        name: str,
        identifier: str,
        split = (0.8, 0.1, 0.1),
        split_seed = 1234,
    ):
        self.name = name
        "Human-readable name"

        self.identifier = identifier
        "Unique, machine-readable identifier"

        self.split = split
        "Ratios of train/test/val split"

        self.split_seed = split_seed
        "Change this to split different data into train/test/val sets"

    @abc.abstractmethod
    def configure(self, task: Task):
        pass

    @abc.abstractmethod
    def load(self, force_cache_reload: bool) -> DSets:
        """
        Load the dataset and adapt for the configured task. Implementations
        should cache and snapshot the dataset to disk, unless this is a very
        small dataset, or there is some other reason not to. This can be done
        using the `snapshot_cache_split` function.
        """
        pass

    def _snapshot_cache_split(self, d: tf.data.Dataset, buffer_size=None) -> DSets:
        """
        Split a dataset into train, val, and test sets, and snapshot the train set to a cache.

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
        d = d.snapshot(f"./_cache/tf/{self.identifier}", compression=None)
        d = d.shuffle(buffer_size=buffer_size, seed=self.split_seed)

        # default split is 80/10/10
        test_size = int(n * self.split[1])
        val_size = int(n * self.split[2])
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
        self._adaptor: Callable[[DSets], DSets] | None = None
    
    def recieve_dataset_config(self, cfg):
        assert isinstance(cfg, self.ds_config_type), f"Expected {self.ds_config_type}, got {type(cfg)}"
        self.ds_cfg = cfg
    
    def recieve_model_config(self, cfg):
        assert isinstance(cfg, self.model_config_type), f"Expected {self.model_config_type}, got {type(cfg)}"
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
    
    @abc.abstractmethod
    def make_final_layer(self) -> tf.keras.layers.Layer:
        pass
    
    @abc.abstractmethod
    def make_loss_fn(self) -> Callable[..., tf.Tensor]:
        pass

    @abc.abstractmethod
    def make_predict_fn(self) -> Callable:
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
        assert isinstance(cfg, self.task_config_type), f"Expected {self.task_config_type}, got {type(cfg)}"
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
        assert isinstance(cfg, self.embd_cfg_type), f"Expected {self.embd_cfg_type}, got {type(cfg)}"
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

    def __init__(self, dataset: MxDataset, task: Task, embedding: Embedding, model: MxModel, identifier: str) -> None:
        
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
        self.model = model
        
        # check identifier is machine-friendly
        assert re.match(r"^[a-z][a-z0-9_\-]+$", identifier), f"'{identifier}' is not a valid identifier, use only a-z, 0-9, _, -"
        
        self.identifier = identifier

    def get_model(self) -> Model:
        
        embedder = self.embedding.make_embedder()
        model = self.model.make_model()
        final_layer = self.task.make_final_layer()

        return Model(
            inputs=embedder.inputs,
            outputs=final_layer(model(embedder.outputs)),
            name=type(self.embedding).__name__ + '-' + type(self.model).__name__
        )

    def get_loss_fn(self) -> Callable[..., tf.Tensor]:
        return self.task.make_loss_fn()
    
    def get_train_data(self, batch_size: int, n_steps: int, force_cache_reload: bool = False) -> tf.data.Dataset:

        dsets = self.dataset.load(force_cache_reload=force_cache_reload)
        dsets = self.task.process(dsets)

        ds_train = dsets.train.map(lambda x: {
            "inputs": x["inputs"],
            "targets": x["targets"],
            # omit "extra" in training
        })
        
        if not self.task.does_batching:
            ds_train = ds_train.batch(batch_size)
        
        ds_train = (
            ds_train
            .take(n_steps)
            .enumerate()
            .map(lambda i, x: (i, (x["inputs"], x["targets"])))
        )

        return ds_train


    
