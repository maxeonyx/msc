from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import time
from typing import Callable

import enlighten

from ._train import MyMetric

def metric_bar_format(metrics: list[MyMetric]):

    if len(metrics) == 0:
        return " ¦ {fill} ¦ ", " ¦ {fill} ¦ "

    headers = [(f"{m.name}", f" ({m.unit})" if m.unit else "") for m in metrics]
    width = max(len(title) + len(unit) for title, unit in headers)
    width = max(width, 7)
    values = [
        m.result().numpy()
        for m in metrics
    ]
    values = [
        # format to 4 significant digits, aligned to the decimal point
        f"{val: > {width}.4g}"
        for val in values
    ]
    headers = [f"{title + unit:>{width}}" for title, unit in headers]

    headers = f" ¦ {{fill}}| {' | '.join(headers)} |{{fill}} ¦ "
    values  = f" ¦ {{fill}}| {' | '.join(values )} |{{fill}} ¦ "
    return headers, values

SPINNER = "⠁⠂⠄⡀⢀⠠⠐⠈"

@dataclass(frozen=False)
class Progress:
    manager: enlighten.Manager
    run_name: str
    min_delta: float = 0.5 / 8
    indent_level: int = 0
    tasks: list[str] = field(default_factory=list)
    update_fns: set[Callable] = field(default_factory=set)

    def __post_init__(self) -> None:

        title_format = " #----- {fill} {task_format} {fill} -----# "
        self.title_bar=self.manager.status_bar(status_format=title_format, task_format=self.task_format())

    def task_format(self):
        if len(self.tasks) >= 1:
            tasks = ": " + " > ".join(self.tasks)
        else:
            tasks = ""

        return f"\"{self.run_name}\"{tasks}"

    @contextmanager
    def enter_spinner(self, name: str, desc: str):
        indent_level = self.indent_level
        indent = "    " * self.indent_level

        def update(i):
            spinner_bar.update(spinner=SPINNER[i % len(SPINNER)], indent=indent, desc=desc)
        
        status_format = "{indent} {spinner} {desc}"
        try:
            with self.manager.status_bar(status_format=status_format, desc=desc, spinner=SPINNER[0], indent=indent, min_delta=self.min_delta, leave=False) as spinner_bar:
                self.indent_level += 1
                self.tasks.append(name)
                    
                self.update_fns.add(update)

                yield
        finally:
            self.indent_level = indent_level
            if update in self.update_fns:
                self.update_fns.remove(update)
            if len(self.tasks) > 0 and self.tasks[-1] == name:
                self.tasks.pop()

    @contextmanager
    def enter_progbar(self, total: int, name: str, desc: str, unit: str ='steps'):
        indent_level = self.indent_level
        indent = "    " * self.indent_level
        
        bar_desc = f"{indent}{desc}"
        try:
            with self.manager.counter(total=total, desc=bar_desc, indent=indent, unit=unit, min_delta=self.min_delta, leave=False) as prog_bar:
                self.indent_level += 1
                self.tasks.append(name)

                yield prog_bar
        finally:
            self.indent_level = indent_level
            if len(self.tasks) > 0 and self.tasks[-1] == name:
                self.tasks.pop()
    
    @contextmanager
    def enter_training(self, n_epochs: int, metrics: list[MyMetric]):
        name = "Train"
        metrics_update = None

        header_fmt, value_fmt = metric_bar_format([])

        try:
            with self.manager.status_bar(status_format=header_fmt, min_delta=self.min_delta, leave=True) as metric_header_bar,\
                self.manager.status_bar(status_format=value_fmt, min_delta=self.min_delta, leave=True) as metric_value_bar,\
                self.enter_progbar(total=n_epochs, name=name, desc="Overall Progress", unit="epochs") as prog_bar:

                def metrics_update(i_step):
                    headers, values = metric_bar_format(metrics)
                    metric_header_bar.update(status_format=headers)
                    metric_value_bar.update(status_format=values)
                    
                self.update_fns.add(metrics_update)

                yield prog_bar
        finally:
            if metrics_update is not None and metrics_update in self.update_fns:
                self.update_fns.remove(metrics_update)
        



@contextmanager
def create_progress_manager(
    run_name: str,
):


    t = None
    update_title_bar = None
    try:
        with enlighten.get_manager() as manager:
            prog = Progress(
                manager=manager,
                run_name=run_name,
            )

            done = False
            def update():
                nonlocal done
                i = 0
                while not done:
                    print(f"in update {i}")
                    for update_fn in prog.update_fns:
                        update_fn(i)
                    i += 1
                    time.sleep(prog.min_delta)
            
            def update_title_bar(i):
                prog.title_bar.update(task_format=prog.task_format())
            
            prog.update_fns.add(update_title_bar)

            t = threading.Thread(None, update).start()

            yield prog

    finally:
        done = True
        if t is not None:
            t.join()
        if update_title_bar is not None and update_title_bar in prog.update_fns:
            prog.update_fns.remove(update_title_bar)

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
