from dataclasses import dataclass, field
import math
import pprint
from typing import Callable, Literal, Set, Union

from mx import utils, datasets as ds
from mx.datasets import orig_tasks
from mx.progress import Progress
from ._metrics import *

@dataclass(frozen=True)
class TrainLoopCfg:
    """
    Config for the training loop.
    """

    optimizer: Literal["adam", "sgd"] = "adam"

    n_steps_per_epoch: int = 500

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
    Wraps a training step function. This is used to update metrics.
    
    Args:
    :param model: The model to train.
    :param data: The data to train on.
    """

    train_step = make_train_step(model, optimizer, loss_fn)

    def train_step_wrapper(data):
        i_step, (inputs_batch, targets_batch) = data
        outputs_batch, loss = train_step(inputs_batch, targets_batch)

        metric_inputs = {
            "loss": loss,
            "step": i_step,
            "targets": targets_batch,
            **outputs_batch,
            **inputs_batch,
        }
        
        for metric in metrics:
            metric.update(metric_inputs)
        
    return train_step_wrapper

def default_metrics(loss_fn) -> dict[str, list[MxMetric]]:
    """
    Default metrics.
    """
    return {
        "epoch_metrics": [
            RunningMean(TimeSinceLastCall(reset_every_epoch=False), name="epoch_time", dtype=tf.float64),
        ],
        "train_step_metrics": [
            TimeSinceLastCall(name="step_time"),
            RunningMean(fn=loss_fn, unit=None, name="loss_epoch"),
            InstantaneousMetric(fn=loss_fn, unit=None, name="loss"),
            Rolling(length=100, fn=loss_fn, unit=None, name="loss_100"),
            Rolling(length=1000, fn=loss_fn, unit=None, name="loss_1000"),

        ],
        "val_step_metrics": [
            TimeSinceLastCall(name="eval_step_time"),
            RunningMean(fn=loss_fn, unit=None, name="eval_loss_epoch"),
            InstantaneousMetric(fn=loss_fn, unit=None, name="eval_loss"),
            Rolling(length=100, fn=loss_fn, unit=None, name="eval_loss_100"),
            Rolling(length=1000, fn=loss_fn, unit=None, name="eval_loss_1000"),
        ],
    }

def make_train_loop(train_cfg: TrainLoopCfg, task_cfg: orig_tasks.TaskCfg, run_name: str, task: ds.DSet, loss_fn: Callable, model: Model, metrics: Union[Literal["default"], dict[str, list[MxMetric]], tuple[Literal["default plus"], dict[str, list[MxMetric]]]] = "default", make_train_step: Callable = default_make_train_step):
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
    elif isinstance(metrics, tuple) and metrics[0] == "default plus":
        def_metrics = default_metrics(loss_fn)
        metrics = def_metrics | metrics[1]
    elif isinstance(metrics, dict):
        pass 
    else:
        raise ValueError(f"Metrics was invalid. Must be a dict with keys 'epoch_metrics', 'train_step_metrics', and 'val_step_metrics'. Got: {metrics}")

    ds_train = task.train

    def train_loop(prog: Progress):

        n_steps = ds_train.cardinality().numpy()
        i_all_step = 0
        n_epochs = math.ceil(n_steps / train_cfg.n_steps_per_epoch)
        try:
            try:
                with prog.enter_spinner(name="Init data", desc="Generating first data..."):
                    steps = iter(ds_train)
                    data = next(steps)
            except StopIteration:
                raise ValueError("No data in dataset")
            
            with prog.enter_spinner(name="Compile train loop", desc="Compiling train loop..."):
                train_step_wrapper = make_train_step_wrapper(
                    model,
                    optimizer,
                    loss_fn,
                    metrics["train_step_metrics"],
                    make_train_step)
                train_step_wrapper = tf.function(train_step_wrapper)
                train_step_wrapper(data)
                i_all_step += 1

            train_loop_metrics = metrics["train_step_metrics"] + metrics["epoch_metrics"]
            with prog.enter_training(n_epochs, train_loop_metrics) as train_prog_bar:
                i_epoch = 0
                while i_epoch < n_epochs:
                    if i_epoch == 0:
                        start_at = 1 # because we already did the first training step when compiling the train loop
                    else:
                        start_at = 0

                    if (i_epoch+1) * train_cfg.n_steps_per_epoch > n_steps:
                        n_steps_this_epoch = n_steps - i_epoch * train_cfg.n_steps_per_epoch
                    else:
                        n_steps_this_epoch = train_cfg.n_steps_per_epoch
                    
                    is_last_epoch = i_epoch == (n_epochs - 1)

                    with prog.enter_progbar(total=n_steps_this_epoch, name=f"Epoch {i_epoch}", desc=f"Epoch {i_epoch}", start_at=start_at, delete_if_success=not is_last_epoch) as epoch_prog_bar:
                        for m in metrics["epoch_metrics"]:
                            m.update({"epoch": i_epoch, "step": i_all_step})
                        i_step = start_at
                        while i_step < n_steps_this_epoch:
                            try:
                                data = next(steps)
                            except StopIteration:
                                print("WARNING: Ran out of data before epoch was finished. This is probably a bug.")
                                return
                            train_step_wrapper(data)
                            i_step += 1
                            i_all_step += 1
                            epoch_prog_bar.update()
                        
                    i_epoch += 1
                    train_prog_bar.update()
        except KeyboardInterrupt:
            print("User interrupted training.")
    
    return train_loop
