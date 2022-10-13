from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import isnan, pi
import sys
import threading
import time
import traceback
from typing import Callable

import enlighten

from mx.export import export
from mx.run_name import get_run_name

## Note: we don't import using mx.prelude or mx.utils because we want this file
## to be usable before tensorflow is imported.

from mx.metrics import MxMetric

SPINNER_1X2 = "⠁⠂⠄⡀⢀⠠⠐⠈"
SPINNER_2X2 = ["⠁ ", "⠂ ", "⠄ ", "⡀ ", "⢀ ", " ⡀", " ⢀", " ⠠", " ⠐", " ⠈", " ⠁", "⠈ "]
SPINNER = SPINNER_2X2
CROSS = "⭕"
TICK = "💚"

VERTICAL_BAR = '│'
HORIZONTAL_BAR = '─'
LEFT_CORNER = "╭"
TOP_TEE = "┬"
LEFT_STOP = "╼"
RIGHT_STOP = "╾"
RIGHT_CORNER = "╮"

def metric_bar_format(metrics: dict[str, MxMetric]):
    metrics = metrics.values()

    if len(metrics) == 0:
        empty_format = f"{VERTICAL_BAR} {{fill}}"
        return empty_format, empty_format

    headers = [(f"{m.name}", f" ({m.unit})" if m.unit else "") for m in metrics]
    metric_formats = [f"{{metric_{m.name}}}" for m in metrics]
    header_bar_format = f"{VERTICAL_BAR}{{fill}}{LEFT_CORNER}{f'{TOP_TEE}'.join(metric_formats)}{RIGHT_CORNER}{{fill}}"
    value_bar_format = f"{VERTICAL_BAR}{{fill}}{VERTICAL_BAR}{f'{VERTICAL_BAR}'.join(metric_formats)}{VERTICAL_BAR}{{fill}}"
    return header_bar_format, value_bar_format

def metric_format(metrics_dict: dict[str, MxMetric]):
    metrics = metrics_dict.values()

    if len(metrics) == 0:
        empty_format = f"{VERTICAL_BAR} {{fill}}"
        return empty_format, empty_format

    def value_format(m: MxMetric, width):

        val = m.result()

        width_nopad = width - 2
        if m.fmt is not None:
            s = m.fmt.format(val)
        elif val is None:
            s = f"{' ... ':^{width_nopad}}"
        elif isnan(val):
            s = f"{' NaN ':^{width_nopad}}"
        elif isinstance(val, int):
            s = f"{val:> {width_nopad}}"
        elif isinstance(val, str):
            s = f"{val:<{width_nopad}}"
        else: # assume float or float-like
            s = f"{val:> {width_nopad}.3g}"

        return f" {s} "

    def header_format(title, unit, width):
        return f"{ LEFT_STOP + ' ' + title + unit + ' ' + RIGHT_STOP :{HORIZONTAL_BAR}^{width}}"

    len_of_stoppers_and_gap = 4
    max_len_of_numeric_val = 13
    headers = [(f"{m.name}", f" ({m.unit})" if m.unit else "") for m in metrics]
    widths = [max(len(title) + len(unit) + len_of_stoppers_and_gap, max_len_of_numeric_val) for title, unit in headers]

    headers = [(f"{m.name}", f" ({m.unit})" if m.unit else "") for m in metrics]
    metric_names = [f"metric_{m.name}" for m in metrics]

    return {
        name: header_format(title, unit, width) for name, width, (title, unit) in zip(metric_names, widths, headers)
    }, {
        name: value_format(m, width) for name, width, m in zip(metric_names, widths, metrics)
    }

@export
@dataclass(frozen=False)
class Progress:
    manager: enlighten.Manager
    run_name: str
    min_delta: float = 0.5 / 8
    indent_level: int = 0
    tasks: list[str] = field(default_factory=list)
    update_fns: list[Callable[[int], None]] = field(default_factory=list)
    update_fns_to_remove = []

    def __post_init__(self) -> None:

        title_format = LEFT_CORNER + "{fill}{task_format}{fill}" + LEFT_STOP
        self.title_bar=self.manager.status_bar(status_format=title_format, task_format=self.task_format(), fill=HORIZONTAL_BAR)

    def task_format(self):

        task_str = " > ".join(self.tasks)

        if self.run_name is not None:
            run_str = f"Run {self.run_name}"
        else:
            run_str = ""

        if len(task_str) > 0 and len(run_str) > 0:
            return f" {run_str}: {task_str} "
        elif len(task_str) > 0:
            return f" {task_str} "
        elif len(run_str) > 0:
            return f" {run_str} "
        else:
            return ""

    @contextmanager
    def enter_spinner(self, name: str, desc: str, delete_on_success=False):
        indent_level = self.indent_level
        indent = "    " * self.indent_level

        status_format = VERTICAL_BAR + " {indent}{spinner} {desc}"
        spinner_bar = self.manager.status_bar(status_format=status_format, desc=desc, spinner=SPINNER[0], indent=indent, min_delta=self.min_delta, leave=True)

        state = "running"
        closed = False
        def update(i):
            if not closed:
                if state == "running":
                    spinner = SPINNER[i % len(SPINNER)]
                elif state == "success":
                    spinner = TICK
                else:
                    spinner = CROSS
                spinner_bar.update(spinner=spinner, desc=desc)

        try:
            self.indent_level += 1
            self.tasks.append(name)

            self.update_fns.append(update)

            yield

            if delete_on_success:
                closed = True
                spinner_bar.leave = False
                spinner_bar.close()
            else:
                desc += " done."
                state = "success"
        except Exception as e:
            state = "error"
            raise e
        finally:
            self.indent_level = indent_level
            if update in self.update_fns:
                def close():
                    if not closed:
                        spinner_bar.close()
                self.update_fns_to_remove.append((update, close))
            if len(self.tasks) > 0 and self.tasks[-1] == name:
                self.tasks.pop()

    @contextmanager
    def enter_progbar(self, total: int | None, name: str, desc: str, unit: str ='steps', start_at: int = 0, delete_on_success: bool = True):
        indent_level = self.indent_level
        indent = "    " * self.indent_level

        counter_format = VERTICAL_BAR + ' {indent}{spinner} {desc}{desc_pad}{count:d}{unit}{unit_pad}{fill}[ {elapsed}, {rate:.2f}{unit_pad}{unit}/s]'
        bar_format = VERTICAL_BAR + ' {indent}{spinner} {desc}{desc_pad}{percentage:3.0f}% |{bar}| {count:{len_total}d}/{total:d} {unit} [ {elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s ]'

        prog_bar = self.manager.counter(total=total, spinner=SPINNER[0], desc=desc, indent=indent, unit=unit, min_delta=self.min_delta, bar_format=bar_format, counter_format=counter_format, count=start_at, leave=True)

        state = "running"
        closed = False
        def update(i):
            if not closed:
                if state == "running":
                    spinner = SPINNER[i % len(SPINNER)]
                elif state == "success":
                    spinner = TICK
                else:
                    spinner = CROSS
                prog_bar.update(incr=0, spinner=spinner)

        try:
            self.indent_level += 1
            self.tasks.append(name)

            self.update_fns.append(update)

            yield prog_bar

            if delete_on_success:
                closed = True
                prog_bar.leave = False
                prog_bar.close()
            else:
                state = "success"
        except Exception as e:
            state = "error"
            raise e
        finally:
            self.indent_level = indent_level
            if update in self.update_fns:
                def close():
                    if not closed:
                        prog_bar.close()
                self.update_fns_to_remove.append((update, close))
            if len(self.tasks) > 0 and self.tasks[-1] == name:
                self.tasks.pop()

    @contextmanager
    def enter_training(self, n_epochs: int, metrics: list[MxMetric]):
        name = "Train"
        metrics_update = None

        header_fmt, value_fmt = metric_bar_format(metrics=metrics)
        headers, values = metric_format(metrics)

        try:
            metric_header_bar = self.manager.status_bar(status_format=header_fmt, min_delta=self.min_delta, leave=True, **headers)
            metric_value_bar = self.manager.status_bar(status_format=value_fmt, min_delta=self.min_delta, leave=True, **values)

            def metrics_update(i_step):
                headers, values = metric_format(metrics)
                metric_header_bar.update(**headers)
                metric_value_bar.update(**values)

            self.update_fns.append(metrics_update)

            with self.enter_progbar(total=n_epochs, name=name, desc="Overall Progress", unit="epochs", delete_on_success=False) as prog_bar:

                yield prog_bar

        finally:
            if metrics_update is not None and metrics_update in self.update_fns:
                def close():
                    metric_header_bar.close()
                    metric_value_bar.close()
                self.update_fns_to_remove.append((metrics_update, close))


manager = None

@export
@contextmanager
def create_progress_manager(
    run_name: str | None = None,
):
    global manager
    t = None
    update_title_bar = None
    with enlighten.get_manager() as e_manager:
        try:
            manager = Progress(
                manager=e_manager,
                run_name=run_name,
            )

            done = False
            def update():
                nonlocal done
                i = 0
                while not done:
                    l = len(manager.update_fns)
                    for j in range(l):
                        manager.update_fns[j](i)

                    for (f, close) in manager.update_fns_to_remove:
                        if f in manager.update_fns:
                            manager.update_fns.remove(f)
                            close()
                    i += 1
                    time.sleep(manager.min_delta)

            def update_title_bar(i):
                manager.title_bar.update(task_format=manager.task_format())

            manager.update_fns.append(update_title_bar)

            t = threading.Thread(None, update).start()

            yield manager
        except Exception as e:
            traceback.print_exc()
        finally:
            sys.stderr.flush()
            sys.stdout.flush()
            time.sleep(manager.min_delta)
            done = True
            if t is not None:
                t.join()
            if update_title_bar is not None and update_title_bar in manager.update_fns:
                manager.update_fns_to_remove.append((update_title_bar, lambda: manager.title_bar.close()))
            manager = None

if __name__ == '__main__':
    with create_progress_manager("Test") as prog:
        with prog.enter_spinner("Loading", "Loading Data"):
            time.sleep(2)
        with prog.enter_training(3, []) as prog_bar:
            for i in prog_bar(range(3)):
                with prog.enter_progbar(3, f"Epoch {i}", f"Epoch {i}") as epoch_bar:
                    for j in epoch_bar(range(3)):
                        time.sleep(0.1)
        with prog.enter_progbar(10, "Visualize", "Visualizing Data") as prog_bar:
            for i in prog_bar(range(10)):
                time.sleep(0.1)


@export
@contextmanager
def spinner(desc: str):

    if manager is None:
        with create_progress_manager() as prog:
            with prog.enter_spinner(desc, desc) as spinner:
                yield spinner
    else:
        with manager.enter_spinner(desc, desc) as spinner:
            yield spinner

def init_with_progress():
    run_name = get_run_name()
    with create_progress_manager(run_name) as pm:
        with pm.enter_spinner("Init Tensorflow", "Initializing Tensorflow..."):
            import mx.prelude
