from dataclasses import dataclass, field
from typing import Literal, Set, Union

from mx import utils, datasets as ds
from mx.datasets import tasks
from mx.progress import Progress
from ._metrics import *

@dataclass(frozen=True)
class TrainLoopCfg:
    """
    Config for the training loop.
    """

    optimizer: Literal["adam", "sgd"] = "adam"

    checkpoint_interval: Union[int, Literal["epoch"], Literal["never"]] = "epoch"
    log_interval: Union[int, Literal["epoch"], Literal["never"]] = "10"
    log_type: Set[Literal["tensorboard", "wandb"]] = frozenset({"tensorboard"})


def default_train_step(model: Model, optimizer: keras.optimizers.Optimizer, inputs: dict[str, TensorLike], targets: TensorLike, loss: Callable) -> tuple[TensorLike, TensorLike]:
    """
    Default training step.
    """
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = loss(targets, outputs)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return outputs, loss

def train_step_wrapper(model: Model, optimizer: tf.keras.optimizers.Optimizer, data: tf.data.Dataset, loss: Callable, metrics: Set[MxMetric], train_step: Callable):
    """
    Wraps a training step function.
    
    Args:
    :param model: The model to train.
    :param data: The data to train on.
    """
    # loop over fused steps
    for i_step, (inputs_batch, targets_batch) in data:
        outputs_batch, loss = train_step(model, inputs_batch, targets_batch, loss)

        for metric in metrics:
            metric.__call__(i_step, inputs_batch | targets_batch | outputs_batch | { "loss": loss })

def default_metrics(loss_fn) -> list[MxMetric]:
    """
    Default metrics.
    """
    return (
        [
            TimeSinceLastCall(name="step_time"),
            RunningMean(TimeSinceLastCall(name="epoch_time", reset_every_epoch=False)),
            RunningMean(fn=loss_fn, unit=None, name="loss_epoch"),
            InstantaneousMetric(fn=loss_fn, unit=None, name="loss"),
            Rolling(length=100, fn=loss_fn, unit=None, name="loss_100"),
            Rolling(length=1000, fn=loss_fn, unit=None, name="loss_1000"),
        ],
        [
            TimeSinceLastCall(name="eval_step_time"),
            RunningMean(fn=loss_fn, unit=None, name="eval_loss_epoch"),
            InstantaneousMetric(fn=loss_fn, unit=None, name="eval_loss"),
            Rolling(length=100, fn=loss_fn, unit=None, name="eval_loss_100"),
            Rolling(length=1000, fn=loss_fn, unit=None, name="eval_loss_1000"),
        ],
    )

def make_train_loop(train_cfg: TrainLoopCfg, task_cfg: tasks.TaskCfg, run_name: str, task: ds.DSet, loss_fn: Callable, model: Model, metrics: Union[Literal["default"], tuple[list[MxMetric], list[MxMetric]], tuple[Literal["default plus"], list[MxMetric], list[MxMetric]]] = "default", train_step: Callable = default_train_step):
    """
    Make the training loop.
    """

    # make optimizer
    if train_cfg.optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam()
    elif train_cfg.optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD()
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg.optimizer}")
    
    if metrics == "default":
        metrics = default_metrics(loss_fn)
    elif isinstance(metrics, tuple) and len(metrics) == 3 and metrics[0] == "default plus":
        def_metrics, def_eval_metrics = default_metrics(loss_fn)
        metrics = (
            def_metrics + metrics[1],
            def_eval_metrics + metrics[2],
        )
    elif isinstance(metrics, tuple) and len(metrics) == 2:
        pass # metrics is already a tuple of metrics
    else:
        raise ValueError(f"metrics must be a tuple of train metrics and eval metrics")

    ds_test, ds_val, ds_train = task.destructure()
    n_epochs = ds_train.cardinality()

    def train_loop(prog: Progress):

        with prog.enter_training(n_epochs, metrics[0]) as train_prog_bar:
            for i_epoch, epoch in enumerate(train_prog_bar(ds_train)):
                epoch_steps = epoch.cardinality()
                with prog.enter_counter("Epoch", total=epoch_steps) as epoch_prog_bar:
                    for i_fusedstep, data in epoch_prog_bar(epoch):

                        data = data.map(lambda x: (x["inputs"], x["targets"]))

                        if i_fusedstep == 0:
                            with prog.enter_spinner("Compiling train loop"):
                                train_step = tf.function(train_step_wrapper)
                                train_step(model, optimizer, data, loss_fn, metrics, train_step)
                        else:
                            train_step(model, optimizer, data, loss_fn, metrics, train_step)
    
    return train_loop
