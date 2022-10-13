from dataclasses import dataclass
import math
from os import PathLike
from pathlib import Path
from typing import Callable, Literal, NamedTuple, Set, Union

from contextlib import ExitStack

from mx.prelude import *

from mx.pipeline import Pipeline
from mx.progress import Progress, create_progress_manager
from mx.metrics import MxMetric, RunningMean, Rolling, TimeSinceLastCall, InstantaneousMetric, wrap_loss_fn_for_metrics
from mx.tf_types import NestedTensor
from mx.visualizer import Visualizer, VizCfg

TrainStepReturn = tuple[tft.NestedTensor, tft.Tensor, list[tft.Tensor]]
TrainStepFn = Callable[[tft.NestedTensor, tft.NestedTensor], TrainStepReturn]

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
def default_metrics(loss_fn, n_steps) -> dict[str, dict[str, MxMetric]]:
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
            InstantaneousMetric(name="step", fn=lambda inputs: inputs["step"], dtype=tf.int64, reset_every_epoch=False, fmt=f"{{}}/{n_steps}"),
            RunningMean(fn=loss_fn, name="loss_epoch"),
            InstantaneousMetric(fn=loss_fn, name="loss", reset_every_epoch=False),
            Rolling(length=100, fn=loss_fn, name="loss_100", reset_every_epoch=False),
            Rolling(length=1000, fn=loss_fn, name="loss_1000", reset_every_epoch=False),
        ]),
        "val_step_metrics": list_to_dict([
            TimeSinceLastCall(name="eval_step_time"),
            RunningMean(fn=loss_fn, name="eval_loss_epoch"),
            InstantaneousMetric(fn=loss_fn, name="eval_loss"),
            Rolling(length=100, fn=loss_fn, name="eval_loss_100"),
            Rolling(length=1000, fn=loss_fn, name="eval_loss_1000"),
        ]),
    }

class TrainLoopError(Exception):

    def __init__(self, message, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = message


def get_epoch(i_step: int, epoch_sizes: list[int]):

    i_epoch = 0
    for epoch_size in epoch_sizes:
        if i_step < epoch_size:
            break
        i_step -= epoch_size
        i_epoch += 1

    return i_epoch

def get_remaining_steps_in_epoch(epoch_sizes, i_epoch, i_global_step):
    return epoch_sizes[i_epoch] - (i_global_step - sum(epoch_sizes[:i_epoch]))

class TrainingStuff(NamedTuple):
    model: Model
    loss_fn: Callable
    data: Dataset
    vizr: Visualizer
    train_loop: Callable


@export
def build_for_training(pipeline: Pipeline) -> tuple[Model, typing.Callable[..., tft.Tensor], Visualizer, Callable]:

    run_name = u.get_run_name()

    if run_name is None:
        output_dir = Path("_outputs") / pipeline.name
    else:
        output_dir = Path("_outputs") / run_name / pipeline.name

    model = pipeline.get_model()
    loss_fn = pipeline.get_loss_fn()
    data = pipeline.get_train_data()
    vizr = pipeline.get_visualizer(
        output_dir=output_dir / "viz",
        viz_batch_size = 3,
        viz_cfgs = {
            "bvh_imgs": {
                "render_on": ["start", "epoch"],
                "show_on": ["start", "end", "interrupt"]
            },
        },
    )

    train_loop = make_train_loop(
        pipeline=pipeline,
        run_name=run_name,
        output_dir=output_dir,
    )

    return (
        model,
        loss_fn,
        data,
        vizr,
        train_loop,
    )

@export
def make_train_loop(
    pipeline: Pipeline,
    run_name: str,
    output_dir: PathLike,
    viz_cfgs: dict[str, VizCfg] = {},
    viz_batch_size: int = None,
    optimizer: Literal["adam", "sgd"] | tf.keras.optimizers.Optimizer = "adam",
    n_steps_per_epoch: int | str = "exponential",
    checkpoint_interval: Union[int, Literal["epoch"], Literal["never"]] = "epoch",
    log_interval: Union[int, Literal["epoch"], Literal["never"]] = 10,
    log_type: Set[Literal["tensorboard", "wandb"]] = {"tensorboard"},
    metrics: Union[
        Literal["default"],
        dict[str, dict[str, MxMetric]],
    ] = "default",
    make_train_step: Callable = default_make_train_step,
) -> Callable[[Progress], None]:
    """
    Make the training loop.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = pipeline.get_model()
    loss_fn = pipeline.get_loss_fn()

    # make epoch sizes
    if n_steps_per_epoch == "exponential":
        epoch_sizes = [ e for e in u.exponential_up_to(pipeline.n_steps) ]
    else:
        epoch_sizes = [ e for e in u.constant_up_to(pipeline.n_steps, chunk=n_steps_per_epoch) ]

    # make optimizer
    if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam()
        elif optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD()
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer}")

    # make metrics
    wrapped_loss_fn = wrap_loss_fn_for_metrics(loss_fn)
    if metrics == "default":
        metrics = default_metrics(wrapped_loss_fn, n_steps=pipeline.n_steps)
    elif isinstance(metrics, dict):
        assert "epoch_metrics" in metrics and "train_step_metrics" in metrics and "val_step_metrics" in metrics, "Metrics must be a dict with keys 'epoch_metrics', 'train_step_metrics', and 'val_step_metrics'."
    else:
        raise ValueError(f"Metrics was invalid. Must be a dict with keys 'epoch_metrics', 'train_step_metrics', and 'val_step_metrics'. Got: {metrics}")


    # The following variables are immutable but are initialized
    # on the first call to train_loop. The boolean flags are
    # separate to support errors. The user can change the
    # implementations and use automatic reloading.
    created_initial_checkpoint = False
    created_data_iterator = True
    created_train_step = False

    def train_loop(
        # number of steps to run for. If None, run forever.
        n_steps: int | None = None,
        profile: bool = False,
        eager: bool = False,
        viz_cfgs: dict[str, VizCfg] = viz_cfgs,
        viz_batch_size: int = viz_batch_size,
        pm: Progress = None,
        log_type: Set[Literal["tensorboard", "wandb"]] = log_type,
        metrics: dict[str, dict[str, MxMetric]] = metrics,
    ):
        nonlocal created_initial_checkpoint, created_data_iterator, created_train_step
        # The following variables are/have state.
        # They are here and not simply anonymously in the outer scope
        # so that they can be accessed by the caller
        if not hasattr(train_loop, "i_step"):
            train_loop.i_step = 0
        if not hasattr(train_loop, "i_epoch"):
            train_loop.i_epoch = 0
        if not hasattr(train_loop, "data_iterator"):
            train_loop.data_iterator = None
        if not hasattr(train_loop, "checkpoints"):
            train_loop.checkpoints = []
        if not hasattr(train_loop, "optimizer"):
            train_loop.optimizer = optimizer
        if not hasattr(train_loop, "train_step_fn"):
            train_loop.train_step_fn = None
        if not hasattr(train_loop, "checkpointer"):
            train_loop.checkpointer = None

        # always re-create the vizualizer, so that it can be reloaded
        vizr = pipeline.get_visualizer(output_dir=output_dir, viz_cfgs=viz_cfgs, viz_batch_size=viz_batch_size)

        # run eagerly if requested
        # todo: currently this is not working
        tf.config.run_functions_eagerly(eager)

        with ExitStack() as stack:
            if pm is None:
                pm = stack.enter_context(create_progress_manager(run_name=run_name))

            try:

                train_loop.checkpointer = tf.train.Checkpoint(model, optimizer=optimizer)

                if "tensorboard" in log_type:
                    tb_writer = tf.summary.create_file_writer(str(output_dir / "logs"))
                else:
                    tb_writer = tf.summary.create_noop_writer()

                with pm.enter_spinner("Saving model", "Saving initial model..."):
                    train_loop.checkpoints.append(train_loop.checkpointer.save(output_dir / "checkpoints"))

                if not created_data_iterator or not created_train_step:
                    try:
                        with pm.enter_spinner(name="Init data", desc="Generating first data..."):
                            dataset = pipeline.get_train_data()
                            # compile
                            initial_data = next(iter(dataset))
                            train_loop.data_iterator = iter(dataset)
                        created_data_iterator = True
                    except StopIteration:
                        raise ValueError("No data in dataset")

                with tb_writer.as_default():

                    vizr.before_train_start(step=train_loop.i_step, pm=pm)

                    try:
                        if profile:
                            tf.profiler.experimental.start(str(output_dir / "profile"))

                        train_loop_metrics = metrics["train_step_metrics"] | metrics["epoch_metrics"]

                        if not created_train_step:
                            with pm.enter_spinner(name="Compile train loop", desc="Compiling train loop..."):
                                train_loop.train_step_fn = make_train_step_wrapper(
                                    model,
                                    optimizer,
                                    loss_fn,
                                    metrics["train_step_metrics"],
                                    make_train_step,
                                    tb_writer,
                                )
                                train_loop.train_step_fn = tf.function(train_loop.train_step_fn)
                                do_log = log_interval != "never"
                                train_loop.train_step_fn(initial_data, do_log)
                                train_loop.checkpointer.restore(train_loop.checkpoints[-1])

                            created_train_step = True

                        def run_for(n_steps, is_last_epoch):
                            nonlocal created_train_step

                            last_epoch_loss = None
                            with pm.enter_progbar(total=n_steps, name=f"Epoch {train_loop.i_epoch}", desc=f"Epoch {train_loop.i_epoch}", delete_on_success=not is_last_epoch) as epoch_prog_bar:

                                i_step = 0
                                while i_step < n_steps:

                                    with tf.profiler.experimental.Trace("train", step_num=train_loop.i_step, _r=1):
                                        try:
                                            data = next(train_loop.data_iterator)
                                        except StopIteration:
                                            raise TrainLoopError("WARNING: Ran out of data before epoch was finished. This is probably a bug with your data generation code. Exiting training loop.")

                                        do_log = (
                                            (log_interval == "epoch" and i_step == 0) or
                                            (type(log_interval) == int and i_step % log_interval == 0)
                                        ) # always false if log_interval == "never"

                                        train_loop.train_step_fn(data, do_log)

                                    if type(checkpoint_interval) == int and train_loop.i_step % checkpoint_interval == 0:
                                        with pm.enter_spinner("Checkpoint", "Checkpointing weights...", delete_on_success=True):
                                            train_loop.checkpoints.append(train_loop.checkpointer.save(output_dir / "checkpoints"))

                                    i_step += 1
                                    train_loop.i_step += 1
                                    # statelessly set i_epoch
                                    train_loop.i_epoch = get_epoch(train_loop.i_step, epoch_sizes)
                                    epoch_prog_bar.count = i_step
                                    epoch_prog_bar.refresh()

                                vizr.on_epoch_end(step=train_loop.i_step, pm=pm)

                            for _, m in metrics["epoch_metrics"].items():
                                    m.update({"epoch": train_loop.i_epoch, "step": train_loop.i_step})

                            if "loss_epoch" in train_loop_metrics:
                                epoch_loss_metric = train_loop_metrics["loss_epoch"]
                                epoch_loss = epoch_loss_metric.result()
                                if last_epoch_loss is not None:
                                    if epoch_loss > last_epoch_loss:
                                        raise TrainLoopError("WARNING: Epoch *training* loss increased from last epoch. This is probably a bug in your model. Exiting training loop.")
                                last_epoch_loss = epoch_loss

                            # reset metrics
                            for _, m in train_loop_metrics.items():
                                if m.reset_every_epoch:
                                    m.reset()

                            if checkpoint_interval == "epoch":
                                with pm.enter_spinner("Checkpoint", "Checkpointing weights...", delete_on_success=True):
                                    train_loop.checkpointer.save(output_dir / "checkpoints")
                        #############


                        if n_steps is None:
                            with pm.enter_training(len(epoch_sizes), train_loop_metrics) as train_prog_bar:
                                while train_loop.i_epoch < len(epoch_sizes):
                                    n_steps = get_remaining_steps_in_epoch(epoch_sizes, train_loop.i_epoch, train_loop.i_step)
                                    if n_steps == 0:
                                        raise TrainLoopError("WARNING: No steps left. Exiting training loop.")
                                    run_for(n_steps, train_loop.i_epoch == (len(epoch_sizes) - 1))
                                    train_prog_bar.count = train_loop.i_epoch
                                    train_prog_bar.refresh()
                        else:
                            run_for(n_steps, True)


                    finally:
                        if profile:
                            with pm.enter_spinner("Save profiler data", "Saving data from performance profiling..."):
                                tf.profiler.experimental.stop(save=True)

                    vizr.after_train_end(pm=pm, step=train_loop.i_step+1)
            except TrainLoopError as e:
                print(e.message)
            except KeyboardInterrupt:
                vizr.on_interrupt(pm=pm, step=train_loop.i_step)
                print("User interrupted training.")

    return train_loop
