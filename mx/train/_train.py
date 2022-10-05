from dataclasses import dataclass, field
import pprint
from typing import Callable, Literal, Set, Union

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


def default_make_train_step(model: Model, optimizer: keras.optimizers.Optimizer, loss_fn: Callable) -> tuple[TensorLike, TensorLike]:
    """
    Factory function for default training step.
    """
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = loss_fn({ "targets": targets, **outputs })
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return outputs, loss
    
    return train_step

def make_train_step_wrapper(model: Model, optimizer: tf.keras.optimizers.Optimizer, loss_fn: Callable, metrics: Set[MxMetric], make_train_step: Callable):
    """
    Wraps a training step function. This is used to make a fused training step, and also
    to update metrics.
    
    Args:
    :param model: The model to train.
    :param data: The data to train on.
    """

    train_step = make_train_step(model, optimizer, loss_fn)

    def train_step_wrapper(data):
        # loop over fused steps
        for i_step, (inputs_batch, targets_batch) in data:
            outputs_batch, loss = train_step(inputs_batch, targets_batch)

            metric_inputs = {
                "loss": loss,
                "step": i_step,
                "targets": targets_batch,
                **outputs_batch,
                **inputs_batch,
            }
            pprint.pp(metric_inputs)
            for metric in metrics:
                metric.update(metric_inputs)
    return train_step_wrapper

def default_metrics(loss_fn) -> list[MxMetric]:
    """
    Default metrics.
    """
    return (
        [
            TimeSinceLastCall(name="step_time"),
            RunningMean(TimeSinceLastCall(name="epoch_time", reset_every_epoch=False), dtype=tf.float64),
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

def make_train_loop(train_cfg: TrainLoopCfg, task_cfg: tasks.TaskCfg, run_name: str, task: ds.DSet, loss_fn: Callable, model: Model, metrics: Union[Literal["default"], tuple[list[MxMetric], list[MxMetric]], tuple[Literal["default plus"], list[MxMetric], list[MxMetric]]] = "default", make_train_step: Callable = default_make_train_step):
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
        train_metrics, eval_metrics = default_metrics(loss_fn)
    elif isinstance(metrics, tuple) and len(metrics) == 3 and metrics[0] == "default plus":
        def_metrics, def_eval_metrics = default_metrics(loss_fn)
        train_metrics, eval_metrics = (
            def_metrics + metrics[1],
            def_eval_metrics + metrics[2],
        )
    elif isinstance(metrics, tuple) and len(metrics) == 2:
        pass # metrics is already a tuple of metrics
    else:
        raise ValueError(f"metrics must be a tuple of train metrics and eval metrics")

    ds_train = task.train

    def train_loop(prog: Progress):

        print(ds_train.element_spec)

        n_epochs = ds_train.cardinality().numpy()
        with prog.enter_training(n_epochs, train_metrics) as train_prog_bar:
            for i_epoch, epoch in ds_train:
                epoch_steps = epoch.cardinality().numpy()
                with prog.enter_progbar(total=epoch_steps, name=f"Epoch {i_epoch}", desc="Epoch") as epoch_prog_bar:
                    for i_fusedstep, data in epoch:
                        data = data.map(lambda i, x: (i, (x["inputs"], x["targets"])))
                        if i_fusedstep == 0:
                            with prog.enter_spinner(name="Compile train loop", desc="Compiling train loop..."):
                                train_step_wrapper = make_train_step_wrapper(
                                    model,
                                    optimizer,
                                    loss_fn,
                                    train_metrics,
                                    make_train_step)
                                train_step_wrapper = tf.function(train_step_wrapper)
                                train_step_wrapper(data)
                        else:
                            train_step_wrapper(data)
                        epoch_prog_bar.update()
                train_prog_bar.update()
    
    return train_loop
