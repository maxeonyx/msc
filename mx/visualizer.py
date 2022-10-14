import abc
from os import PathLike

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

    def __init__(self, name: str, desc: str):
        self.name = name
        self.desc = desc

    @abc.abstractmethod
    def render(self, output_location, timestep=None):
        """
        Make a visualization, save it, but don't show it.
        """
        pass

    @abc.abstractmethod
    def _get_uri(self, output_location) -> str:
        """
        Get the URI of the visualization.
        """
        pass

    def __call__(self, output_location, timestep=None):
        """
        Make a visualization, save it, and show it.
        """
        self.render(output_location, timestep)
        self.show(output_location)

    def show(self, output_location):
        import webbrowser
        uri = self._get_uri(output_location)
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

    def __init__(self, name: str, desc: str):
        self.name = name
        self.desc = desc

    @abc.abstractmethod
    def make_and_save_visualization(self, do_update, output_location, timestep=None):
        pass

    def render(self, output_location, timestep=None):
        """
        Create and save a visualization. Update the held visualization. Don't show it.
        """
        self.make_and_save_visualization(do_update=True, output_location=output_location, timestep=timestep)

    def __call__(self, output_location, timestep=None):
        """
        Create and save a visualization. Don't update the held visualization. Show it.
        For interactive usage.
        """
        self.make_and_save_visualization(do_update=False, output_location=output_location, timestep=timestep)
        self.show(output_location)


@export
class HoloMapVisualization(StatefulVisualization, abc.ABC):
    """
    A visualization that maintains a figure and uses HoloMaps to display the data.

    The HoloMap has dimensions "timestep" and "batch" and is updated with
    the data returned by the "fn" function.
    """

    def __init__(self, name: str, desc: str):
        super().__init__(
            name=name,
            desc=desc,
        )
        self._fig = None

    @abc.abstractmethod
    def make_hmaps(self, timestep) -> list[hv.HoloMap]:
        """
        Implements the figure.
        """
        pass

    def _filename(self, output_location) -> Path:
        f = u.ensure_suffix(Path(output_location), ".html")
        f.parent.mkdir(parents=True, exist_ok=True)
        return f

    def make_and_save_visualization(self, do_update: bool, output_location: PathLike, timestep=None):
        """
        Display figure to screen. (Note: Also saves to file.)
        """
        hmaps = self.make_hmaps(timestep)
        fig = hv.Layout(hmaps).cols(1).opts(
            shared_axes=False,
            title=self.name,
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

        filename = self._filename(output_location)
        hv.save(fig, filename, fmt='html', backend='bokeh')

    def _get_uri(self, output_location) -> str:
        filename = self._filename(output_location)
        return filename.absolute().as_uri()

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
        configs: dict[str, VizCfg],
        output_dir: PathLike,
    ):
        assert isinstance(visualizations, dict), "visualizations must be a dict"

        for k, viz in visualizations.items():
            assert isinstance(viz, Visualization), f"Visualizations must be of type Visualization. Got type of visualizations['{k}'] = {type_name(viz)}"

        self.visualizations = visualizations
        self.set_cfgs(configs)

        self.output_dir = Path(output_dir)

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

    def _do(self, event: str, timestep, pm: Progress):
        for viz, cfg in zip(self.visualizations.values(), self.cfgs.values()):
            if event in cfg["render_on"]:
                with pm.enter_spinner(
                    name=f"Visualize '{viz.name}'",
                    desc=f"Visualizing '{viz.desc}'...",
                    delete_on_success=True
                ):
                    viz.render(self.output_dir / viz.name, timestep=timestep)

            if event in cfg["show_on"]:
                # show the visualization
                viz.show(self.output_dir / viz.name)

    def before_train_start(self, step, pm: Progress):
        self._do("start", timestep=step, pm=pm)

    def after_train_end(self, step, pm: Progress):
        self._do("end", timestep=step, pm=pm)

    def on_epoch_end(self, step, pm: Progress):
        self._do("epoch", timestep=step, pm=pm)

    def on_interrupt(self, step, pm: Progress):
        self._do("interrupt", timestep=step, pm=pm)
