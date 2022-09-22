from dataclasses import dataclass
from typing import Literal, Set, Union

import enlighten
from box import Box as box

from mx import utils, tasks, datasets as ds
from ._metrics import *

@dataclass(frozen=True)
class TrainCfg:
    """
    Config for the training loop.
    """
    batch_size: int = 32
    n_steps: int = 5000
    steps_per_epoch: int = 500
    fused_steps: int = 1

    max_test_steps: int = 100
    test_batch_size: int = 32

    optimizer: Literal["adam", "sgd"] = "adam"

    checkpoint_interval: Union[int, Literal["epoch"], Literal["never"]] = "epoch"
    log_interval: Union[int, Literal["epoch"], Literal["never"]] = "10"
    log_type: Set[Literal["tensorboard", "wandb"]] = frozenset({"tensorboard"})

    metrics: Set[MetricCfg] = frozenset({
        RollingAvgMetricCfg(type="loss", n_steps=100, reset_every_epoch=False),
        InstantaneousMetricCfg(type="total_epoch_time"),
        RollingAvgMetricCfg(type="step_time", n_steps=10),
    })
    eval_metrics: Set[MetricCfg] = frozenset({
        RunningAvgMetricCfg(type="loss"),
        InstantaneousMetricCfg(type="total_epoch_time"),
        RollingAvgMetricCfg(type="step_time", n_steps=10),
    })

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

def train_step_wrapper(model: Model, optimizer: tf.keras.optimizers.Optimizer, data: tf.data.Dataset, loss: Callable, metrics: Set[MyMetric], train_step: Callable):
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
            metric(i_step, inputs_batch | targets_batch | outputs_batch | { "loss": loss })


def make_train_loop(train_cfg: TrainCfg, task_cfg: tasks.TaskCfg, run_name: str, task: ds.DSet, loss_fn: Callable, model: Model, train_step: Callable = default_train_step):
    """
    Make the training loop.
    """
    ds_test, ds_val, ds_train = task.destructure()

    # make optimizer
    if train_cfg.optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam()
    elif train_cfg.optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD()
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg.optimizer}")
    
    # make metrics
    metrics = set()
    for metric_cfg in train_cfg.metrics:
        m = make_metric(metric_cfg, loss_fn=loss_fn)
        metrics.add(m)
    
    # make eval metrics
    eval_metrics = set()
    for metric_cfg in train_cfg.eval_metrics:
        m = make_metric(metric_cfg, loss_fn=loss_fn)
        eval_metrics.add(m)

    if not task_cfg.batch:
        ds_train = ds_train.batch(train_cfg.batch_size)
        ds_val = ds_val.batch(train_cfg.test_batch_size)
        ds_test = ds_test.batch(train_cfg.test_batch_size)
    ds_train = ds_train.take(train_cfg.n_steps).enumerate()
    ds_train = ds_train.window(train_cfg.fused_steps)
    ds_train = ds_train.window(train_cfg.steps_per_epoch//train_cfg.fused_steps)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    n_epochs = train_cfg.n_steps // train_cfg.steps_per_epoch

    # need import here because of circular imports
    from ._prog import ProgressManager

    def train_loop(prog: ProgressManager):

        with prog.enter_training(n_epochs, metrics) as train_prog_bar:
            for i_epoch, epoch in enumerate(train_prog_bar(ds_train)):
                with prog.enter_counter("Epoch", total=train_cfg.steps_per_epoch) as epoch_prog_bar:
                    for i_step, data in epoch_prog_bar(epoch):

                        if i_step == 0:
                            with prog.enter_spinner("Compiling train loop"):
                                train_step = tf.function(train_step_wrapper)
                                train_step(model, optimizer, data, loss_fn, metrics, train_step)
                        else:
                            train_step(model, optimizer, data, loss_fn, metrics, train_step)
    
    return train_loop
