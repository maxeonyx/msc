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

VizCfg = dict[str, list[Literal["start", "end", "epoch", "exit"]]]
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
        - update_on: "start", "end", "twice", or "epoch"
            When to run this visualization.
        - render_on: "start", "end", "twice", or "epoch"
            When to save this visualization.

    Frequency values:

    "start" => before training

    "end" => after training

    "twice" => before and after training

    "epoch" => after each epoch
    """

    def __init__(self,
        visualizations: dict[str, tuple[Visualization, VizCfg]],
        output_dir: PathLike,
    ):
        assert isinstance(visualizations, dict), "visualizations must be a dict"

        for k, val in visualizations.items():
            assert isinstance(val, tuple), f"visualization['{k}'] must be a tuple of (Visualization, cfg). Got visualization['{k}'] = {val}"
            viz, cfg = val
            assert isinstance(viz, Visualization), f"Visualizations must be of type Visualization. Got {viz}"
            assert isinstance(cfg, dict), f"Configs must be of type dict. Got cfgs['{k}']={cfg}"

            assert "render_on" in cfg, f"Visualizations must have a 'render_on' key in their config. Got cfgs['{k}']={cfg}"
            assert "show_on" in cfg, f"Visualizations must have a 'show_on' key in their config. Got cfgs['{k}']={cfg}"

        self.visualizations = visualizations

        self.output_dir = Path(output_dir)

    def _do(self, viz: Visualization, cfg, event: str, timestep, pm: Progress):
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
        for viz, cfg in self.visualizations.values():
            self._do(viz, cfg, "start", timestep=step, pm=pm)

    def after_train_end(self, step, pm: Progress):
        for viz, cfg in self.visualizations.values():
            self._do(viz, cfg, "end", timestep=step, pm=pm)

    def on_epoch_end(self, step, pm: Progress):
        for viz, cfg in self.visualizations.values():
            self._do(viz, cfg, "epoch", timestep=step, pm=pm)

    def on_interrupt(self, step, pm: Progress):
        for viz, cfg in self.visualizations.values():
            self._do(viz, cfg, "exit", timestep=step, pm=pm)
