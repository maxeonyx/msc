from re import A
from this import d
import holoviews as hv

from mx.prelude import *

from mx.pipeline import MxDataset, Task
from mx.tasks import VectorSequenceMSE
from mx.tasks.vector_sequence_mse import RandomSequenceMSE
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
            "image": x["image"],
            "label": x["label"],
            "idxs": u.multidim_indices([28, 28]),
        })

        return dsets

    def configure(self, task: Task):

        if isinstance(task, VectorSequenceMSE) or isinstance(task, RandomSequenceMSE):
            task.recieve_dataset_config(task.ds_config_cls(
                seq_dims=[28, 28],
                n_input_dims=1, # grayscale
                already_batched=True,
            ))
            def adapt_in(x):

                values = x["image"]
                # to float in [-1, 1]
                values = (tf.cast(values, u.dtype()) / 255.) * 2. - 1.
                # flatten x y into a single sequence dimension
                values = ein.rearrange(values, f"... w h c -> ... (w h) c")

                seq_idxs = x["idxs"]
                # remove channel dim from indices
                seq_idxs = seq_idxs[..., :, :2]

                new_x = {
                    "values": values,
                    "seq_idxs": seq_idxs,
                    "extra": {
                        "image": x["image"],
                    },
                }
                if "label" in x:
                    new_x["extra"]["label"] = x["label"]
                return new_x
            self.adapt_in = adapt_in
            def adapt_out(x):

                image = x["values"]
                # from float in [-1, 1] to uint8 in [0, 255]
                image = ((image + 1.) / 2.) * 255.
                image = tf.clip_by_value(image, 0., 255.)
                image = tf.cast(image, tf.uint8)
                # unflatten x y from a single sequence dimension
                image = ein.rearrange(image, f"... (w h) c -> ... w h c", w=28)

                # adapt output from a task's predict_fn into dataset-specific format
                return {
                    "image": image,
                }
            self.adapt_out = adapt_out
        else:
            raise NotImplementedError(f"Dataset {type_name(self)} does not support Task {type_name(task)}. If using autoreload in IPython, try restarting the interpreter.")

        assert self.adapt_in is not None, "Forgot to set self.adapt_in"
        assert self.adapt_out is not None, "Forgot to set self.adapt_out"

    def get_visualizations(self, viz_batch_size, task_specific_predict_fn=None) -> dict[str, Visualization]:

        if task_specific_predict_fn is not None:
            assert self.adapt_in is not None, "To visualize predictions, must call configure() before get_visualizations()"
            assert self.adapt_out is not None, "To visualize predictions, must call configure() before get_visualizations()"

        data = next(iter(
            self.load(force_cache_reload=False)
            .test
            .batch(viz_batch_size)
        ))

        return u.list_to_dict([
            MNISTImageViz(
                data=data,
                predict=(
                    task_specific_predict_fn,
                    self.adapt_in,
                    self.adapt_out,
                ),
            )
        ])


class MNISTImageViz(HoloMapVisualization):
    """
    Produce a HoloMap to display predicted MNIST images.
    """

    def __init__(
        self,
        data,
        predict = (None, None, None),
        name="mnist_images",
        desc="MNIST images",
        output_dir=None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            output_dir=output_dir,
        )

        # validate data in "dataset" format (the output of MxMNIST.load)
        u.validate(data, "data", {
            "image": tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.uint8),
        })
        self._data = data

        if predict is not None:
            # if this visualization will be predicting stuff, idxs need to be
            # provided
            u.validate(data, "data", {
                "image": tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.uint8),
                "idxs": tf.TensorSpec(shape=(None, 784, 2), dtype=tf.int32),
            })

        self._task_predict_fn, self._adapt_in, self._adapt_out = predict

    def _make_hmaps(self, timestep=None) -> list:

        data = self._data

        def img(data, title):
            return hv.Image(data.numpy()).opts(
                title=title,
                cmap="gray",
                aspect='equal',
            )

        def key(batch, timestep):

            if tf.is_tensor(timestep) and timestep.dtype == tf.string:
                timestep = u.tf_str(timestep)
            elif tf.is_tensor(timestep) and timestep.dtype == tf.int32:
                timestep = timestep.numpy()

            if timestep is None:
                return (batch,)
            elif batch is None:
                return (timestep,)
            else:
                return (batch, timestep)

        def title(batch, timestep):

            if tf.is_tensor(timestep) and timestep.dtype == tf.string:
                timestep = u.tf_str(timestep)
            elif tf.is_tensor(timestep):
                timestep = timestep.numpy()

            if batch is None:
                batch = ""
            else:
                batch = f" {batch}"

            if timestep is None:
                return f"Image{batch}"
            elif isinstance(timestep, str):
                return f"Image{batch} @ {timestep}"
            else:
                return f"Image{batch} @ step={timestep}"

        hmaps = []

        key_dims = [
            hv.Dimension(("batch", "Batch")),
        ]
        hmaps.append(hv.GridSpace(
            {
                (i_batch): img(data["image"][i_batch], f"Example {i_batch}")
                for i_batch in range(len(data["image"]))
            },
            kdims=key_dims,
            label="Examples"
        ))

        if self._task_predict_fn is not None:
            task_predict_fn = self._task_predict_fn
            adapt_in = self._adapt_in
            adapt_out = self._adapt_out

            # predict blind. Make a batch of 784 predictions, one for each pixel
            # in the image, each with a specific target idx.
            empty_seq = {
                "values": tf.zeros([784, 0, 1], dtype=u.dtype()),
                "seq_idxs": ein.rearrange(u.multidim_indices([28, 28]), 'hw i -> hw 1 i'),
            }
            pred_blind = task_predict_fn(empty_seq, output_len=1)
            pred_blind = adapt_out({
                "values": ein.rearrange(pred_blind["values"], '(h w) 1 c -> 1 (h w) c', h=28, w=28)
            })

            if timestep is not None:
                key_dims = [hv.Dimension(("timestep", "Timestep"), default=timestep)]
                hmaps.append(
                    hv.HoloMap(
                        {
                            key(None, timestep): img(pred_blind["image"][0], title(None, timestep))
                        },
                        kdims=key_dims,
                        label="Predicted Image (blind)",
                    )
                )
            else:
                hmaps.append(
                    img(pred_blind["image"][0], title(None, None))
                )

            pred = adapt_out(
                task_predict_fn(
                    inputs=adapt_in(data),
                    seed_len=784//2,
                    output_len=784,
                )
            )
            u.stats(pred["image"])

            key_dims = [
                hv.Dimension(("batch", "Batch")),
            ]
            if timestep is not None:
                key_dims.append(hv.Dimension(("timestep", "Timestep"), default=timestep))

            hmaps.extend([
                hv.GridSpace(
                    {
                        key(i_batch, timestep): img(pred_data[i_batch], title(i_batch, timestep))
                        for i_batch in range(len(pred_data))
                    },
                    kdims=key_dims,
                    label=f"Predicted Images ({name})",
                )
                for name, pred_data in pred.items()
            ])


        return hmaps

if __name__ == '__main__':
    u.set_debug()
    mnist = MxMNIST()
    data = next(iter(mnist.load().test.batch(10)))
    tp(data, "MxMNIST data format")
    viz = MNISTImageViz(data)
    viz()
