from re import A
import holoviews as hv

from mx.prelude import *

from mx.pipeline import MxDataset, Task
from mx.tasks import VectorSequenceMSE
from mx.utils import DSets
from mx.visualizer import HoloMapVisualization, Visualization

@export
class MxMNIST(MxDataset):
    """
    Config for the MNIST dataset pipeline.

    Adapts huggingface MNIST to my custom dataset
    interface.
    """

    def __init__(
        self,
        name="mnist",
        desc="MNIST dataset",
        train_val_split=(50000./60000., 10000./60000.),
        split_seed=1234,
    ):
        super().__init__(
            name=name,
            desc=desc,
        )

        self.train_val_split = train_val_split
        "Ratios of train/val split"

        self.split_seed = split_seed
        "Change this to split different data into train/test/val sets"

    def load(self, force_cache_reload=False):
        """
        Load the dataset from disk.
        """
        mnist = tfds.load("mnist", shuffle_files=True)

        train_val = mnist["train"]
        test = mnist["test"]

        n = train_val.cardinality().numpy()
        train_val = train_val.shuffle(n, seed=self.split_seed)
        n_val = int(n * self.train_val_split[1])
        n_train = n - n_val
        train = train_val.take(n_train)
        val = train_val.skip(n_train)

        dsets = DSets(
            train=train,
            val=val,
            test=test,
        )

        dsets = dsets.map(lambda x: {
            **x,
            "idxs": u.multidim_indices_of(x["image"]),
            # add "extra"
            "extra": None,
        })

        dsets = dsets.cache()

        return dsets

    def configure(self, task: Task):

        if isinstance(task, VectorSequenceMSE):
            task.recieve_dataset_config(task.ds_config_type(
                seq_dims=[28, 28],
                n_input_dims=1, # grayscale
            ))
            def adapt_in(x):

                values = x["image"]
                # to float in [0, 1]
                values = tf.cast(values, tf.float32) / 255.
                # flatten x y into a single sequence dimension
                values = ein.rearrange(values, f"... w h c -> ... (w h) c")

                seq_idxs = x["idxs"]
                # remove channel dim from indices
                seq_idxs = seq_idxs[..., :, :2]

                return {
                    "values": values,
                    "seq_idxs": seq_idxs,
                    "extra": x["extra"],
                }
            self.adapt_in = adapt_in
            def adapt_out(x):

                image = x["values"]
                # to uint8 in [0, 255]
                image = tf.cast(tf.clip_by_value(image * 255., 0., 255.), tf.uint8)
                # unflatten x y from a single sequence dimension
                image = ein.rearrange(image, f"... (w h) c -> ... w h c", w=28)

                # adapt output from a task's predict_fn into dataset-specific format
                return {
                    ("image", "Predicted Image"): image,
                }
            self.adapt_out = adapt_out
        else:
            raise NotImplementedError(f"{type_name(task)} not supported by {type(self)}")

        assert self.adapt_in is not None, "Forgot to set self.adapt_in"
        assert self.adapt_out is not None, "Forgot to set self.adapt_out"

    def get_visualizations(self, viz_batch_size, task_specific_predict_fn=None, run_name: str = None) -> dict[str, Visualization]:

        if task_specific_predict_fn is not None:
            assert self.adapt_in is not None, "To visualize predictions, must call configure() before get_visualizations()"
            assert self.adapt_out is not None, "To visualize predictions, must call configure() before get_visualizations()"

        data = (
            self.load(force_cache_reload=False)
            .test
            .take(viz_batch_size)
        )
        data = [ d for d in data ]

        return u.list_to_dict([
            MNISTImageViz(
                data=data,
                task_predict_fn=task_specific_predict_fn,
                adapt_in=self.adapt_in,
                adapt_out=self.adapt_out,
                run_name=run_name,
            )
        ])


class MNISTImageViz(HoloMapVisualization):
    """
    Produce a HoloMap to display predicted MNIST images.

    >>> data = MxMNIST().load()
    >>> task =
    >>> data = [ x for x in data.test.take(5) ]
    >>> viz = MNISTImageViz(data)
    >>> viz.make_hmaps(i_step=0)
    """

    def __init__(
        self,
        data,
        task_predict_fn,
        adapt_in,
        adapt_out,
        name="mnist_images",
        desc="MNIST images",
        run_name: str = None,
        output_dir: str = None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            run_name=run_name,
            output_dir=output_dir,
        )
        self._data = data
        self._task_predict_fn = task_predict_fn
        self._adapt_in = adapt_in
        self._adapt_out = adapt_out

    def _make_hmaps(self, timestep=None):
        data = self._data
        # data in "dataset" format (the output of MxMNIST.load)

        assert isinstance(data, list), f"Expected data to be a list, got {type_name(data)}"
        batch_size = len(data)

        assert isinstance(data[0], dict), f"Expected data to be a list of dicts, got {type_name(data[0])}"
        assert "image" in data[0], f"Expected data to have 'image' key, got {data[0].keys()}"
        assert tf.is_tensor(data[0]["image"]), f"Expected data['image'] to be a tf.Tensor, got {type_name(data[0]['image'])}"
        assert data[0]["image"].shape == (28, 28, 1), f"Expected data['image'] to have shape (28, 28, 1), got {data[0]['image'].shape}"

        def img(data, title):
            return hv.Image(data.numpy()).opts(
                title=title,
                cmap="gray",
                aspect='equal',
            )

        hmaps = []

        if self._task_predict_fn is not None:
            task_predict_fn = self._task_predict_fn
            adapt_in = self._adapt_in
            adapt_out = self._adapt_out

            pred = [
                adapt_out(task_predict_fn(
                    inputs=adapt_in(x),
                    seed_len=784//2,
                    output_len=784,
                ))
                for x in data
            ]

            # use tf.nest.map_structure to stack all component tensors
            pred = tf.nest.map_structure(
                lambda *xs: tf.concat(xs, axis=0),
                *pred,
            )
            key_dims = [
                hv.Dimension(("batch", "Batch")),
            ]
            if timestep is not None:
                key_dims.append(hv.Dimension(("timestep", "Timestep")))

            def key(batch, timestep):

                if tf.is_tensor(timestep) and timestep.dtype == tf.string:
                    timestep = u.tf_str(timestep)
                elif tf.is_tensor(timestep) and timestep.dtype == tf.int32:
                    timestep = timestep.numpy()

                if timestep is None:
                    return (batch,)
                else:
                    return (batch, timestep)

            def title(batch, timestep):

                if tf.is_tensor(timestep) and timestep.dtype == tf.string:
                    timestep = u.tf_str(timestep)
                elif tf.is_tensor(timestep) and timestep.dtype == tf.int32:
                    timestep = timestep.numpy()

                if timestep is None:
                    return f"Image {batch}"
                elif isinstance(timestep, str):
                    return f"Image {batch} @ {timestep}"
                else:
                    return f"Image {batch} @ step={timestep}"

            hmaps.extend([
                hv.GridSpace(
                    {
                        key(i_batch, timestep): img(data[i_batch], title(i_batch, timestep))
                        for i_batch in range(batch_size)
                    },
                    kdims=key_dims,
                    label=predict_type_desc
                )
                for (predict_type_name, predict_type_desc), data in pred.items()
            ])

        key_dims = [
            hv.Dimension(("batch", "Batch")),
        ]
        hmaps.append(hv.GridSpace(
            {
                (i_batch): img(data[i_batch]["image"], f"Example {i_batch}")
                for i_batch in range(batch_size)
            },
            kdims=key_dims,
            label="Examples"
        ))

        return hmaps





if __name__ == '__main__':
    u.set_debug()
    mnist = MxMNIST()
    data = mnist.load()
    tp(data, "mnist data format")
    viz = mnist.get_visualizations(
        viz_batch_size=5,
        task_specific_predict_fn=None,
    )
    viz["mnist_images"]()
