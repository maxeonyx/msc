import abc
from contextlib import ExitStack
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

        run_name = u.get_run_name()
        if u.get_run_name() is not None:
            self.desc = f"{self.desc} #{run_name}"

        self.output_dir = output_dir

    def output_location(self, output_dir: PathLike = None) -> Path:
        """
        Output location, which is either a directory or a filename
        depending on whether the visualization produces a single file
        or many.
        """

        run_name = u.get_run_name()
        if run_name is None:
            default_output_dir = Path("_outputs")
        else:
            default_output_dir = Path("_outputs") / run_name

        output_dir = Path(
            output_dir
            or self.output_dir
            or default_output_dir
        )
        output_location = output_dir / "viz" / self.name
        output_location.mkdir(parents=True, exist_ok=True)
        return output_location

    @abc.abstractmethod
    def render(self, timestep=None, output_dir=None, pm: Progress = None):
        """
        Make a visualization, save it, but don't show it.
        """
        pass

    @abc.abstractmethod
    def _get_uri(self, output_dir) -> str:
        """
        Get the URI of the visualization.
        """
        pass

    def __call__(self, timestep=None, output_dir=None, pm: Progress = None):
        """
        Make a visualization, save it, and show it.
        """

        self.render(timestep, output_dir, pm=pm)
        self.show(output_dir)

    def show(self, output_dir=None):
        import webbrowser
        uri = self._get_uri(output_dir)
        uri_message = f"""
╭───────────────────────────────────╼
│ Open visualization: {self.desc}
│
│     {uri}
│
╰───────────────────────────────────╼
"""
        print(uri_message.strip())
        if not hasattr(self, 'shown'):
            webbrowser.open_new_tab(uri)
            self.shown = True
        else:
            webbrowser.open(uri)

@export
class StatefulVisualization(Visualization, abc.ABC):
    """
    A (possibly-stateful) visualization.

    make_visualization():
        stateless, returns a visualization

    update_visualization():
        stateful, updates the held visualization, returns none

    """

    def __init__(self, name: str, desc: str, output_dir: PathLike | None):
        super().__init__(name, desc, output_dir)


    @abc.abstractmethod
    def _make_and_save_visualization(self, do_update, timestep=None, output_dir=None, pm:Progress=None):
        pass

    def render(self, timestep=None, output_dir=None, pm: Progress | None = None):
        """
        Create and save a visualization. Update the held visualization. Don't show it.
        """

        self._make_and_save_visualization(do_update=True, timestep=timestep, output_dir=output_dir)

    def __call__(self, timestep=None, output_dir=None, pm:Progress=None):
        """
        Create and save a visualization. Don't update the held visualization. Show it.
        For interactive usage.
        """

        self._make_and_save_visualization(do_update=False, timestep=timestep, output_dir=output_dir, pm=pm)
        self.show(output_dir=output_dir)


@export
class HoloMapVisualization(StatefulVisualization, abc.ABC):
    """
    A visualization that maintains a figure and uses HoloMaps to display the data.

    The HoloMap has dimensions "timestep" and "batch" and is updated with
    the data returned by the "fn" function.
    """

    def __init__(self, name: str, desc: str, output_dir: PathLike | None):
        super().__init__(name, desc, output_dir)
        self._fig: hv.Layout = None

    @abc.abstractmethod
    def _make_hmaps(self, timestep, pm:Progress=None) -> list[hv.HoloMap]:
        """
        Implements the figure.
        """
        pass

    def output_location(self, output_dir=None) -> Path:
        output_location = super().output_location(output_dir)
        return u.ensure_suffix(output_location, ".html")

    def _make_and_save_visualization(self, do_update: bool, timestep=None, output_dir=None, pm:Progress=None):
        """
        Display figure to screen. (Note: Also saves to file.)
        """
        hmaps = self._make_hmaps(timestep, pm=pm)
        fig = hv.Layout(hmaps).cols(1).opts(
            shared_axes=False,
            title=self.desc,
            width=1600,
            height=900,
            sizing_mode="stretch_both",
        )

        if do_update:
            if self._fig is None:
                self._fig = fig
            else:
                for existing_hmap, new_hmap in zip(self._fig, fig):
                    existing_hmap.update(new_hmap)
            fig = self._fig

        filename = self.output_location(output_dir)
        hv.save(fig, filename, fmt='html', backend='bokeh')

    def _get_uri(self, output_dir=None) -> str:
        return self.output_location(output_dir).absolute().as_uri()

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
        output_dir: PathLike = None,
    ):
        assert isinstance(visualizations, dict), "visualizations must be a dict"

        for k, viz in visualizations.items():
            assert isinstance(viz, Visualization), f"Visualizations must be of type Visualization. Got type of visualizations['{k}'] = {type_name(viz)}"

        self.visualizations = visualizations
        self.set_cfgs(configs)

        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = None

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

    def __call__(self, timestep=None, pm: Progress=None, output_dir=None):
        output_dir = output_dir or self.output_dir
        for viz, cfg in zip(self.visualizations.values(), self.cfgs.values()):
            with ExitStack() as stack:
                if pm is not None:
                    stack.push(self._spinner(viz, pm))
                viz(timestep=timestep, output_dir=output_dir, pm=pm)

    def _do(self, event: str, timestep, pm: Progress):
        if tf.is_tensor(timestep):
            timestep = timestep.numpy()

        for viz, cfg in zip(self.visualizations.values(), self.cfgs.values()):
            if event in cfg["render_on"]:
                with self._spinner(viz, pm):
                    viz.render(timestep=timestep, output_dir=self.output_dir, pm=pm)
            if event in cfg["show_on"]:
                # show the visualization
                viz.show(output_dir=self.output_dir)

    def before_train_start(self, step, pm: Progress):
        self._do("start", timestep=step, pm=pm)

    def after_train_end(self, step, pm: Progress):
        self._do("end", timestep=step, pm=pm)

    def on_epoch_end(self, step, pm: Progress):
        self._do("epoch", timestep=step, pm=pm)

    def on_interrupt(self, step, pm: Progress):
        self._do("interrupt", timestep=step, pm=pm)
