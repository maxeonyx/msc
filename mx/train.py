from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Literal, Set, Union

from mx.prelude import *

from mx.progress import Progress
from mx.metrics import MxMetric, RunningMean, Rolling, TimeSinceLastCall, InstantaneousMetric, wrap_loss_fn_for_metrics
from mx.tf_types import NestedTensor
from mx.visualizer import Visualizer

TrainStepReturn = tuple[tft.NestedTensor, tft.Tensor, list[tft.Tensor]]
TrainStepFn = Callable[[tft.NestedTensor, tft.NestedTensor], TrainStepReturn]

@export
def default_make_train_step(
    model: Model,
    optimizer: keras.optimizers.Optimizer,
    loss_fn: Callable,
) -> TrainStepFn:
    """
    Factory function for default training step.
    """
    def train_step(inputs, targets) -> TrainStepReturn:
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = loss_fn(targets, outputs)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return outputs, loss, grads

    return train_step

@export
def make_train_step_wrapper(
    model: Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: Callable,
    metrics: dict[str, MxMetric],
    make_train_step: Callable[..., TrainStepFn],
    tb_writer: tf.summary.SummaryWriter,
):
    """
    Wraps a training step function. This is used to update metrics.

    Args:
    :param model: The model to train.
    :param data: The data to train on.
    """

    train_step = make_train_step(model, optimizer, loss_fn)

    def train_step_wrapper(data, do_log: bool):
        i_step, (inputs_batch, targets_batch) = data

        with tb_writer.as_default(step=i_step):
            outputs_batch, loss, grads = train_step(inputs_batch, targets_batch)

            metric_inputs: NestedTensor = {
                "loss": loss,
                "step": i_step,
                "targets": targets_batch,
                "outputs": outputs_batch,
                "inputs": inputs_batch,
            }

            for metric in metrics.values():
                metric.update(metric_inputs)

            if do_log:
                tf.summary.scalar("loss", loss)
                for grad in grads:
                    tf.summary.histogram("grad", grad)
                    tf.summary.scalar("grad_norm", tf.norm(grad))
                for k, v in outputs_batch.items():
                    tf.summary.histogram(f"outputs/{k}", v)

    return train_step_wrapper

@export
def default_metrics(loss_fn) -> dict[str, dict[str, MxMetric]]:
    """
    Default metrics.
    """

    list_to_dict = lambda x: {item.name: item for item in x}

    return {
        "epoch_metrics": list_to_dict([
            RunningMean(TimeSinceLastCall(), name="epoch_time", dtype=tf.float64, reset_every_epoch=False),
        ]),
        "train_step_metrics": list_to_dict([
            TimeSinceLastCall(name="step_time"),
            RunningMean(fn=loss_fn, unit=None, name="loss_epoch"),
            InstantaneousMetric(fn=loss_fn, unit=None, name="loss", reset_every_epoch=False),
            Rolling(length=100, fn=loss_fn, unit=None, name="loss_100", reset_every_epoch=False),
            Rolling(length=1000, fn=loss_fn, unit=None, name="loss_1000", reset_every_epoch=False),
        ]),
        "val_step_metrics": list_to_dict([
            TimeSinceLastCall(name="eval_step_time"),
            RunningMean(fn=loss_fn, unit=None, name="eval_loss_epoch"),
            InstantaneousMetric(fn=loss_fn, unit=None, name="eval_loss"),
            Rolling(length=100, fn=loss_fn, unit=None, name="eval_loss_100"),
            Rolling(length=1000, fn=loss_fn, unit=None, name="eval_loss_1000"),
        ]),
    }

def exponential_up_to(n, base=2):
    i = 0
    total = 0
    while True:
        e = base ** i
        if total + e > n:
            break
        yield e
        i += 1
        total += e
    yield n - total


def constant_up_to(n, chunk):
    total = 0
    while True:
        if total + chunk > n:
            break
        yield chunk
        total += chunk
    yield n - total

@export
def train_loop(
    prog: Progress,
    data: tf.data.Dataset,
    model: Model,
    loss_fn: Callable,
    pipeline_name: str,
    run_name: str,
    vizr: Visualizer,
    optimizer: Literal["adam", "sgd"] | tf.keras.optimizers.Optimizer = "adam",
    n_steps_per_epoch: int | str = "exponential",
    checkpoint_interval: Union[int, Literal["epoch"], Literal["never"]] = "epoch",
    log_interval: Union[int, Literal["epoch"], Literal["never"]] = 10,
    log_type: Set[Literal["tensorboard", "wandb"]] = {"tensorboard"},
    profile: bool = False,
    metrics: Union[
        Literal["default"],
        dict[str, dict[str, MxMetric]],
    ] = "default",
    make_train_step: Callable = default_make_train_step,
    compile = True,
) -> Callable[[Progress], None]:
    """
    Make the training loop.
    """

    output_dir = Path(f"./_outputs/{pipeline_name}/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # make optimizer
    if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam()
        elif optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD()
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer}")

    wrapped_loss_fn = wrap_loss_fn_for_metrics(loss_fn)
    if metrics == "default":
        metrics = default_metrics(wrapped_loss_fn)
    elif isinstance(metrics, dict):
        assert "epoch_metrics" in metrics and "train_step_metrics" in metrics and "val_step_metrics" in metrics, "Metrics must be a dict with keys 'epoch_metrics', 'train_step_metrics', and 'val_step_metrics'."
    else:
        raise ValueError(f"Metrics was invalid. Must be a dict with keys 'epoch_metrics', 'train_step_metrics', and 'val_step_metrics'. Got: {metrics}")

    n_steps = data.cardinality().numpy()
    i_all_step = 0
    if n_steps_per_epoch == "exponential":
        epoch_sizes = [ e for e in exponential_up_to(n_steps) ]
        n_epochs = len(epoch_sizes)
    else:
        epoch_sizes = [ e for e in constant_up_to(n_steps, chunk=n_steps_per_epoch) ]
        n_epochs = len(epoch_sizes)

    try:

        checkpoint = tf.train.Checkpoint(model, optimizer=optimizer)

        if "tensorboard" in log_type:
            tb_writer = tf.summary.create_file_writer(str(output_dir / "logs"))
        else:
            tb_writer = tf.summary.create_noop_writer()

        with prog.enter_spinner("Saving model", "Saving initial model..."):
            checkpoint.save(output_dir / "checkpoints")

        try:
            with prog.enter_spinner(name="Init data", desc="Generating first data..."):
                steps = iter(data)
                data = next(steps)
        except StopIteration:
            raise ValueError("No data in dataset")

        with tb_writer.as_default():

            vizr.before_train_start(step=i_all_step, pm=prog)

            try:
                if profile:
                    tf.profiler.experimental.start(str(output_dir / "profile"))

                with prog.enter_spinner(name="Compile train loop", desc="Compiling train loop..."):
                    train_step_wrapper = make_train_step_wrapper(
                        model,
                        optimizer,
                        loss_fn,
                        metrics["train_step_metrics"],
                        make_train_step,
                        tb_writer,
                    )
                    if compile:
                        train_step_wrapper = tf.function(train_step_wrapper)
                    do_log = log_interval != "never"
                    train_step_wrapper(data, do_log)
                    i_all_step += 1

                train_loop_metrics = metrics["train_step_metrics"] | metrics["epoch_metrics"]
                with prog.enter_training(n_epochs, train_loop_metrics) as train_prog_bar:

                    i_epoch = 0
                    while i_epoch < n_epochs:
                        if i_epoch == 0:
                            start_at = 1 # because we already did the first training step when compiling the train loop
                        else:
                            start_at = 0

                        n_steps_this_epoch = epoch_sizes[i_epoch]

                        is_last_epoch = i_epoch == (n_epochs - 1)

                        last_epoch_loss = None
                        with prog.enter_progbar(total=n_steps_this_epoch, name=f"Epoch {i_epoch}", desc=f"Epoch {i_epoch}", start_at=start_at, delete_on_success=not is_last_epoch) as epoch_prog_bar:

                            i_step = start_at
                            while i_step < n_steps_this_epoch:

                                with tf.profiler.experimental.Trace("train", step_num=i_all_step, _r=1):
                                    try:
                                        data = next(steps)
                                    except StopIteration:
                                        print("WARNING: Ran out of data before epoch was finished. This is probably a bug with your data generation code. Exiting training loop.")
                                        return

                                    do_log = (
                                        (log_interval == "epoch" and i_step == start_at) or
                                        (type(log_interval) == int and i_step % log_interval == 0)
                                    ) # always false if log_interval == "never"

                                    train_step_wrapper(data, do_log)

                                if type(checkpoint_interval) == int and i_all_step % checkpoint_interval == 0:
                                    with prog.enter_spinner("Checkpoint", "Checkpointing weights...", delete_on_success=True):
                                        checkpoint.save(output_dir / "checkpoints")

                                i_step += 1
                                i_all_step += 1
                                epoch_prog_bar.update()

                            vizr.on_epoch_end(step=i_all_step, pm=prog)

                        for _, m in metrics["epoch_metrics"].items():
                                m.update({"epoch": i_epoch, "step": i_all_step})

                        if "loss_epoch" in train_loop_metrics:
                            epoch_loss_metric = train_loop_metrics["loss_epoch"]
                            epoch_loss = epoch_loss_metric.result()
                            if last_epoch_loss is not None:
                                if epoch_loss > last_epoch_loss:
                                    print("WARNING: Epoch *training* loss increased from last epoch. This is probably a bug in your model. Exiting training loop.")
                                    return
                            last_epoch_loss = epoch_loss

                        # reset metrics
                        for _, m in train_loop_metrics.items():
                            if m.reset_every_epoch:
                                m.reset()

                        if checkpoint_interval == "epoch":
                            with prog.enter_spinner("Checkpoint", "Checkpointing weights...", delete_on_success=True):
                                checkpoint.save(output_dir / "checkpoints")

                        i_epoch += 1
                        train_prog_bar.update()
            finally:
                if profile:
                    with prog.enter_spinner("Save profiler data", "Saving data from performance profiling..."):
                        tf.profiler.experimental.stop(save=True)

            vizr.after_train_end(pm=prog, step=i_all_step+1)

    except KeyboardInterrupt:
        vizr.on_interrupt(pm=prog, step=i_all_step)
        print("User interrupted training.")
