import abc
from contextlib import ExitStack
from datetime import datetime
from os import PathLike
import threading

import holoviews as hv
from mx.progress import Progress
hv.extension('bokeh')

from mx.prelude import *
from mx import utils as u

__all__, export = exporter()


@export
class Visualization(abc.ABC):
    """
    Base class for vizualizations.
    """

    def __init__(self, name: str, desc: str, output_dir: PathLike | None):
        self.name = name
        self.desc = desc
        self.output_dir = output_dir

    @abc.abstractmethod
    def __call__(self, timestep=None, pm: Progress = None, show=True):
        """
        Make a visualization, save it, and show it.
        """
        pass


events = ["start", "end", "interrupt", "epoch"]

VizCfg = dict[str, list[Literal["start", "end", "interrupt", "epoch"]]]
export("VizCfg")

@export
class Visualizer:
    """
    Visualizes outputs from a training run. Given a dict of
    Visualization objects, it will update, display, and save them at
    the appropriate times during the training loop.

    input:
        visualizations: dict of (Visualization, cfg) pairs

    cfg should be a dict with the following keys:
    -   "render_on": list of strings, one of "start", "end", "epoch", "interrupt"
    -   "show_on": list of strings, one of "start", "end", "epoch", "interrupt"

    Event values:

    "start" => before training

    "end" => after training

    "interrupt" => user cancelled training

    "epoch" => after each epoch
    """

    def __init__(self,
        visualizations: dict[str, Visualization],
        configs: dict[str, VizCfg] = {},
    ):
        assert isinstance(visualizations, dict), "visualizations must be a dict"

        for k, viz in visualizations.items():
            assert isinstance(viz, Visualization), f"Visualizations must be of type Visualization. Got type of visualizations['{k}'] = {type_name(viz)}"

        self.visualizations = visualizations
        self.set_cfgs(configs)

    def set_cfgs(self, cfgs: dict[str, VizCfg]):

        vizs = self.visualizations

        assert isinstance(cfgs, dict), f"cfgs must be a dict, got {type_name(cfgs)}"
        for k, cfg in cfgs.items():
            assert k in vizs, f"Vizualization '{k}' not found in visualizations. Available visualizations: {list(vizs.keys())}"
            assert isinstance(cfg, dict), f"cfgs['{k}'] must be a dict, got {type_name(cfg)}"
            assert "render_on" in cfg, f"cfgs['{k}'] must have a 'render_on' key"
            assert "show_on" in cfg, f"cfgs['{k}'] must have a 'show_on' key"

            assert all((e in events) for e in cfg["render_on"]), f"cfgs['{k}']['render_on'] can only be 'start', 'end', 'interrupt', or 'epoch', got {cfg['render_on']}"
            assert all((e in events) for e in cfg["show_on"]), f"cfgs['{k}']['show_on'] can only be 'start', 'end', 'interrupt', or 'epoch', got {cfg['show_on']}"

            # visualizations can't be shown until they have been rendered. Validate that here.
            requires = {
                "start": ["start"],
                "epoch": ["start", "epoch"],
                "end": ["start", "epoch", "end"],
                "interrupt": ["start"],
            }
            for e in events:
                if e in cfg["show_on"]:
                    assert any((r in cfg["render_on"]) for r in requires[e]), f"cfgs['{k}'] cannot be shown before it is rendered. Add one of {requires[e]} to cfgs['{k}']['render_on']"

        # if cfg not specified, use default
        for k in vizs.keys():
            if k not in cfgs:
                cfgs[k] = {
                    ## default config ##
                    "render_on": ["start", "epoch"],
                    "show_on": ["start", "end", "interrupt"],
                }

        self.cfgs = cfgs

    def _spinner(self, viz, pm):
        return pm.enter_spinner(
            name=f"Visualize: {viz.name}",
            desc=f"Creating visualization: {viz.desc} ...",
            delete_on_success=True,
        )

    def __call__(self, timestep=None, pm: Progress=None):
        for viz, cfg in zip(self.visualizations.values(), self.cfgs.values()):
            with ExitStack() as stack:
                if pm is not None:
                    stack.push(self._spinner(viz, pm))
                viz(timestep=timestep, pm=pm)

    def _do(self, event: str, timestep, pm: Progress):
        if tf.is_tensor(timestep):
            timestep = timestep.numpy()

        for viz, cfg in zip(self.visualizations.values(), self.cfgs.values()):
            show = event in cfg["show_on"]
            if event in cfg["render_on"]:
                with self._spinner(viz, pm):
                    viz(timestep=timestep, pm=pm, show=show)

    def before_train_start(self, step, pm: Progress):
        self._do("start", timestep=step, pm=pm)

    def after_train_end(self, step, pm: Progress):
        self._do("end", timestep=step, pm=pm)

    def on_epoch_end(self, step, pm: Progress):
        self._do("epoch", timestep=step, pm=pm)

    def on_interrupt(self, step, pm: Progress):
        self._do("interrupt", timestep=step, pm=pm)
