from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
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

    def adapt(self, dsets: DSets, batch_size, test_batch_size) -> DSets:
        """
        Adapt a dataset from raw format to task-specific task-input format
        """

        assert self.adapt_in is not None, "Must call dataset.configure(task) before dataset.adapt(ds)"

        dsets = dsets.map(self.adapt_in)

        dsets.train.map(lambda x: {
            **x,
            "extra": None,
        })
        dsets.val.map(lambda x: {
            **x,
            "extra": None,
        })

        dsets = dsets.batch(batch_size, test_batch_size)

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
    def get_visualizations(self, viz_batch_size, task_specific_predict_fn, run_name) -> dict[str, Visualization]:
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



@dataclass
class Task_DatasetConfig:
    n_input_dims: int
    "Dimensionality of input vectors (recieved by run from dataset)"

    already_batched: bool
    "Whether the dataset is already batched"

@dataclass
class Task_ModelConfig:
    n_output_embd: int
    "Number of output embedding dimensions (recieved by final layer from model)"

@export
class Task(abc.ABC):


    def __init__(
        self,
        name: str,
        desc: str,
        does_batching: bool = False,
    ):
        self.name = name
        "Unique, machine-readable identifier"

        self.desc = desc
        "Human-readable name"

        self.does_batching = does_batching
        "Whether the task or the dataset does batching"

        self.ds_config_cls: Type[Task_DatasetConfig] = Task_DatasetConfig
        "Required dataset-specific config"

        self.model_config_type: Type[Task_ModelConfig] = Task_ModelConfig
        "Required model-specific config"

        ## Configured by self.recieve_dataset_config(cfg) ##
        self.ds_cfg = None
        ## Configured by self.recieve_model_config(cfg) ##
        self.model_cfg = None
        ## Configured by task.configure(embedding) ##
        self.adapt_in: Callable[[DSets], DSets] | None = None

    def recieve_dataset_config(self, cfg):
        assert isinstance(cfg, self.ds_config_cls), f"Expected {self.ds_config_cls}, got {type_name(cfg)}"
        self.ds_cfg = cfg
        self.does_batching = cfg.already_batched

    def recieve_model_config(self, cfg):
        assert isinstance(cfg, self.model_config_type), f"Expected {self.model_config_type}, got {type_name(cfg)}"
        self.model_cfg = cfg

    @abc.abstractmethod
    def configure(self, cfg, embedding: MxEmbedding):
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

@dataclass
class Embedding_TaskConfig:
    n_input_dims: int
    "Size of the input vector (recieved from the task)."

@export
class MxEmbedding(abc.ABC):
    """
    Base class for embeddings.
    """

    def __init__(
        self,
        desc: str,
        name: str,
        n_embd: int
    ):
        self.desc = desc
        "Human-friendly description"

        self.name = name
        "Unique, machine-friendly name"

        self.n_embd = n_embd
        "Number of embedding dimensions"

        self.task_config_type: Type[Embedding_TaskConfig] = Embedding_TaskConfig

        ## set by self.recieve_task_config(cfg) ##
        self.task_cfg: Embedding_TaskConfig = None

    def receive_task_config(self, cfg: Embedding_TaskConfig):
        assert isinstance(cfg, self.task_config_type), f"Expected {self.task_config_type}, got {type_name(cfg)}"
        self.task_cfg = cfg

    @abc.abstractmethod
    def configure(self, model: MxModel):
        pass

    @abc.abstractmethod
    def make_embedder(self) -> Model:
        pass


@dataclass
class Model_EmbeddingConfig:
    n_embd: int
    """Number of embedding dimensions."""

@export
class MxModel(abc.ABC):

    def __init__(
        self,
        name: str,
        desc: str,
    ):
        self.name = name
        "Machine-friendly name"

        self.desc = desc
        "Human-friendly description"

        self.embd_cfg_type: Type[Model_EmbeddingConfig] = Model_EmbeddingConfig

        ## set by self.recieve_embedding_config(cfg) ##
        self.embd_cfg: Model_EmbeddingConfig = None

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

    def __init__(self, batch_size: int,  n_steps: int, dataset: MxDataset, task: Task, embedding: MxEmbedding, model: MxModel, name: str, desc: str, n_models: int = 1, use_float16: bool = False, test_batch_size: int = None, viz_batch_size: int = 3) -> None:

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.viz_batch_size = viz_batch_size
        self.n_steps = n_steps

        # check identifier is machine-friendly
        assert re.match(r"^[a-zA-Z][a-zA-Z0-9_\-]+$", name), f"'{name}' is not a valid identifier, use only a-z, 0-9, _, -"

        self.name = name
        self.desc = desc

        self.use_float16 = use_float16

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
        self.n_models = n_models

        self._output_dir = None

    def model_name(self, i: int = None):

        assert self.n_models == 1 or (self.n_models > 1 and i is not None), "Must provide i when n_models > 1"

        if i is not None:
            suffix = f"_{i}"
        else:
            suffix = ""

        return type_name(self.embedding) + '_' + type_name(self.mx_model) + suffix

    def new_model(self, i: int = None) -> Model:
        "Create a new model instance, and save the model definition to disk."
        embedder = self.embedding.make_embedder()
        backbone = self.mx_model.make_model()
        final_layer = self.task.make_final_layer()

        model = Model(
            inputs=embedder.inputs,
            outputs=final_layer(backbone(embedder(embedder.inputs))),
            name=self.name,
        )

        model.save(self.output_dir() / model.name)

        return model

    def output_dir(self):

        if self._output_dir is not None:
            return self._output_dir

        run_name = u.get_run_name()

        date = datetime.now().date()
        time = datetime.now().isoformat(timespec='seconds')

        if run_name == "dev":
            run_dir = f"dev-{time}"
        elif run_name is None:
            run_dir = f"interactive-{time}"
        else:
            run_dir = f"{date}-{run_name}"

        if run_name is not None:
            output_dir = Path("_outputs") / run_dir / self.name
        else:
            output_dir = Path("_outputs") / self.name

        output_dir.mkdir(parents=True, exist_ok=True)

        self._output_dir = output_dir
        return output_dir

    def make_or_load_model(self, force_new=False) -> Model:
        "Create a new model, or load the existing one if it exists. Model definitions are saved on creation."

        assert self.n_models == 1, "Can't make_or_load_model() when n_models > 1. Use make_or_load_models() instead."

        if not force_new:
            try:
                return keras.models.load_model(self.output_dir() / self.model_name())
            except OSError:
                pass

        return self.new_model()

    def make_or_load_models(self, force_new=False):
        "Yield models. Load, or create new models. Model definitions are saved on creation."

        assert self.n_models > 1, "Can't make_or_load_models() when n_models == 1. Use make_or_load_model() instead."

        if not force_new:
            try:
                for i in range(self.n_models):
                    yield keras.models.load_model(self.output_dir() / self.model_name(i))
                return
            except OSError:
                pass
        for i in range(self.n_models):
            yield self.new_model(i)

    def make_loss_fn(self) -> Callable[..., tf.Tensor]:
        return self.task.make_loss_fn()

    def make_train_data(self, force_cache_reload: bool = False) -> tf.data.Dataset:

        dsets = self.dataset.load(force_cache_reload=force_cache_reload)
        dsets = self.dataset.adapt(dsets, self.batch_size, self.test_batch_size)
        dsets = self.task.process(dsets)
        dsets = self.task.adapt(dsets)

        d = dsets.train

        d = (
            d
            .take(self.n_steps)
            .enumerate()
            .map(lambda i, x: (i, (x["inputs"], x["targets"])))
        )

        d = d.prefetch(100)

        return d
