from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Callable, Type

from mx.prelude import *
from mx.utils import DSets
from mx.visualizer import Visualization, Visualizer, VizCfg

from mx import datasets as mxd, tasks as mxt, models as mxm, embeddings as mxe

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
    def load(self, batch_size, test_batch_size, force_cache_reload: bool=False) -> DSets:
        """
        Load the dataset and adapt for the configured task. Implementations
        should cache and snapshot the dataset to disk, unless this is a very
        small dataset, or there is some other reason not to. This can be done
        using the `snapshot_cache_split` function.
        """
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

def choose_from_iterator(iterator, title):
    l = list(iterator)
    max_name = max(len(l.name) for l in l)
    max_i = max(len(str(i)) for i in range(len(l)))
    print()
    print(f"    {title}")
    for i, item in enumerate(l):
        default = "(default)" if hasattr(item, 'default') and item.default else " "*9

        print(f"{default} ({i: >{max_i}}): {item.name:.<{max_name}}....{item.desc}")
    choice = input("        choice: ")
    if choice == "":
        choice = next(i for i, item in enumerate(l) if hasattr(item, 'default') and item.default)
    else:
        choice = int(choice)
    return l[choice]


def datasets():
    yield Box(
        default=True,
        name="mnist",
        desc="MNIST",
        opts=[],
    )
    yield Box(
        name="bvh",
        desc="BVH (Dense)",
        opts=[
            [
                Box(
                    name="decimate",
                    desc="Decimate",
                    val=1.5,
                ),
                Box(
                    name="no_decimate",
                    desc="No Decimate",
                    val=False,
                ),
            ],
        ],
    )

def tasks(dataset):
    yield Box(
        name="next",
        desc="Default order (left-to-right, top-to-bottom)",
    )
    yield Box(
        name="rand",
        desc="Random Order",
    )
    if dataset.name.startswith("bvh"):
        yield Box(
            name="targ",
            desc="Next Frame, except the start (targeted)",
        )

def embeddings(ds, task):
    if ds.name == "mnist":
        yield Box(
            default=True,
            name="valcode",
            desc="Dense / Codebook",
            discretize=False,
        )
        yield Box(
            name="codecode",
            desc="Discretized / Codebook",
            discretize=True,
        )
        yield Box(
            name="valsin",
            desc="Dense / Sinusoidal",
            discretize=False,
        )
        yield Box(
            name="codesin",
            desc="Discretized / Sinusoidal",
            discretize=True,
        )
    elif ds.name.startswith("bvh"):
        yield Box(
            default=True,
            name="angsin",
            desc="Angles / Sinusoidal",
        )
        yield Box(
            name="angcode",
            desc="Angles / Codebook",
        )


def models(task):
    yield Box(
        name="mlp",
        desc="2-layer MLP",
        opts=[],
    )
    yield Box(
        default=True,
        name="transformer",
        desc="Decoder-only Transformer (default opts)",
        opts=[
            [
                Box(
                    default=True,
                    name="lma",
                    desc="Use learned-mix-add",
                    val=True,
                ),
                Box(
                    name="no_lma",
                    desc="Normal addition",
                    val=False,
                ),
            ],
            [
                Box(
                    default=True,
                    name="batchnorm",
                    desc="Use batchnorm",
                    val=True,
                ),
                Box(
                    name="no_batchnorm",
                    desc="No batchnorm",
                    val=False,
                ),
            ],
            [
                Box(
                    default=True,
                    name="layer_norm",
                    desc="Use layer norm",
                    val=True,
                ),
                Box(
                    name="no_layer_norm",
                    desc="No layer norm",
                    val=False,
                ),
            ],
        ],
    )
    yield Box(
        name="resnet",
        desc="Resnet",
        opts=[
            [
                Box(
                    default=True,
                    name="lma",
                    desc="Use learned-mix-add",
                    val=True,
                ),
                Box(
                    name="no_lma",
                    desc="Normal addition",
                    val=False,
                ),
            ],
            [
                Box(
                    default=True,
                    name="batchnorm",
                    desc="Use batchnorm",
                    val=True,
                ),
                Box(
                    name="no_batchnorm",
                    desc="No batchnorm",
                    val=False,
                ),
            ],
        ],
    )


def model_sizes():
    yield Box(
        name="tiny",
        desc="Tiny: 2L 32E",
        n_embd=32,
        n_hidden=64,
        n_heads=4,
        n_layers=2,
    )
    yield Box(
        name="small",
        desc="Small: 4L 64E",
        n_embd=64,
        n_hidden=128,
        n_heads=8,
        n_layers=4,
    )
    yield Box(
        default=True,
        name="medium",
        desc="Medium: 6L 128E",
        n_embd=128,
        n_hidden=256,
        n_heads=8,
        n_layers=6,
    )
    yield Box(
        name="large",
        desc="Large: 8L 256E",
        n_embd=256,
        n_hidden=512,
        n_heads=12,
        n_layers=8,
    )
    yield Box(
        name="honking",
        desc="Honker: 16L 512E",
        n_embd=512,
        n_hidden=1024,
        n_heads=12,
        n_layers=16,
    )
    yield Box(
        name="wide",
        desc="Wide: 4L 1024E",
        n_embd=1024,
        n_hidden=2048,
        n_heads=8,
        n_layers=4,
    )
    yield Box(
        name="deep",
        desc="Deep: 24L 256E",
        n_embd=256,
        n_hidden=512,
        n_heads=8,
        n_layers=24,
    )

def lengths():
    yield Box(
        name="debug",
        desc="10 steps",
        n_steps=10,
    )
    yield Box(
        name="normal",
        desc="20k steps",
        n_steps=20_000,
    )
    yield Box(
        default=True,
        name="long",
        desc="1M steps",
        n_steps=1_000_000,
    )

def chunk_sizes(dataset):
    if dataset.name.startswith("mnist"):
        yield Box(
            name="blind",
            desc="Blind",
            chunk_size=1,
            batch_size=784*16,
            n_test_val_repeats=5,
        )
        yield Box(
            default=True,
            name="chunks",
            desc="Chunks",
            chunk_size=28*2,
            batch_size=784*16//(28*2),
            n_test_val_repeats=5,
        )
        yield Box(
            name="fullimg",
            desc="Full Image",
            chunk_size=784,
            batch_size=16,
            n_test_val_repeats=5,
        )
    elif dataset.name.startswith("bvh"):
        yield Box(
            name="chunkshort",
            desc="Chunk Size = 32, Batch Size = 256",
            chunk_size=32,
            batch_size=16*16,
        )
        yield Box(
            default=True,
            name="chunklong",
            desc="Chunk Size = 512, Batch Size = 16",
            chunk_size=32*16,
            batch_size=16,
        )
        yield Box(
            name="chunklittle",
            desc="Chunk Size = 32, Batch Size = 16",
            chunk_size=32,
            batch_size=16,
        )

# choose pipeline interactively
# previous implementation was a big list, now we do it part-at-a-time
# pipeline, dataset, task, model = choose_pipeline()
def choose_pipeline():
    cfg = Box()
    cfg.dataset = choose_from_iterator(datasets(), "Dataset")
    for opt in cfg.dataset.opts:
        cfg.dataset[opt.name] = choose_from_iterator(opt, "Dataset Option")
    cfg.task = choose_from_iterator(tasks(cfg.dataset), "Task")
    cfg.embd = choose_from_iterator(embeddings(cfg.dataset, cfg.task), "Embedding")
    cfg.model = choose_from_iterator(models(cfg.task), "Model")
    for opt in cfg.model.opts:
        choice = choose_from_iterator(opt, "Model Option")
        cfg.model[choice.name] = choice
    cfg.model.size = choose_from_iterator(model_sizes(), "Model Size")
    cfg.train = Box()
    cfg.train.length = choose_from_iterator(lengths(), "Training Length")
    cfg.train.chunk = choose_from_iterator(chunk_sizes(cfg.dataset), "Seq & Batch Size")

    if cfg.dataset.name == "mnist":
        dataset = mxd.MxMNIST()
    elif cfg.dataset.name.startswith("bvh"):
        dataset = mxd.BvhDataset(do_decimate=cfg.dataset.decimate)
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset.name}")

    print()
    print(f"Pipeline: {pipeline.name}")
    print(f"    {pipeline.desc}")
    print()
    return pipeline

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
        embedder, inputs = self.embedding.make_embedder()
        backbone = self.mx_model.make_model()
        final_layer = self.task.make_final_layer()

        if u.regtype == 'bvh':
            vals, embd = embedder(inputs)
            embd = backbone(embd)
            outvals = final_layer(embd)

            # outs = vals + outvals
            outs = outvals
            outs = ein.rearrange(
                outs,
                "... (feat sincos) -> ... feat sincos",
                sincos=2,
            )
        else:
            outs = final_layer(backbone(embedder(inputs)))

        model = Model(
            inputs=inputs,
            outputs=outs,
            name=self.model_name(i),
        )

        # model.save(self.output_dir() / model.name)
        model.was_loaded = False

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
        elif run_name == "blessed":
            run_dir = run_name
        else:
            run_dir = f"{date}-{run_name}"

        if run_name is not None:
            output_dir = Path("_outputs") / run_dir / self.name
        else:
            output_dir = Path("_outputs") / self.name

        output_dir.mkdir(parents=True, exist_ok=True)

        self._output_dir = output_dir
        return output_dir

    def make_or_load_model(self, force_new=False, force_not_new=False) -> Model:
        "Create a new model, or load the existing one if it exists. Model definitions are saved on creation."

        assert self.n_models == 1, "Can't make_or_load_model() when n_models > 1. Use make_or_load_models() instead."

        if not force_new:
            try:
                model = keras.models.load_model(self.output_dir() / self.model_name(), compile=False)
                model.was_loaded = True
            except OSError:
                if force_not_new:
                    raise
                model = self.new_model()
        else:
            model = self.new_model()

        model._name = self.model_name()

        return model

    def make_or_load_models(self, force_new=False):
        "Yield models. Load, or create new models. Model definitions are saved on creation."

        assert self.n_models > 1, "Can't make_or_load_models() when n_models == 1. Use make_or_load_model() instead."

        if not force_new:
            for i in range(self.n_models):
                try:
                    model = keras.models.load_model(self.output_dir() / self.model_name(i))
                    checkpointer = tf.train.Checkpoint(model)
                    checkpointer.restore(self.output_dir() / self.model_name(i) / "checkpoint")
                    model._name = self.model_name(i)
                    model.was_loaded = True
                    yield model
                except OSError:
                    pass
        for i in range(self.n_models):
            yield self.new_model(i)

    def make_loss_fn(self) -> Callable[..., tf.Tensor]:
        return self.task.make_loss_fn()

    def make_train_data(self, force_cache_reload: bool = False) -> tf.data.Dataset:

        dsets = self.dataset.load(self.batch_size, self.test_batch_size, force_cache_reload=force_cache_reload)
        dsets = self.dataset.adapt(dsets)
        dsets = self.task.process(dsets)
        dsets = self.task.adapt(dsets)

        d_train = dsets.train
        d_train = (
            d_train
            .take(self.n_steps)
            .enumerate()
            .map(lambda i, x: (i, (x["inputs"], x["targets"])))
        )
        d_train = d_train.prefetch(100)

        d_val = dsets.val
        d_val = (
            d_val
            .enumerate()
            .map(lambda i, x: (i, (x["inputs"], x["targets"])))
        )
        d_val = d_val.prefetch(100)

        return d_train, d_val
