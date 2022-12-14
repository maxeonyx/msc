from re import A
from this import d
import holoviews as hv
from mx.predict import PredictInputs, PredictOutputs, predict

from mx.prelude import *

from mx.pipeline import MxDataset, Task
from mx.progress import Progress, create_progress_manager
from mx.tasks import RandomTokens, VectorSequenceMSE, RandomSequenceMSE
from mx.utils import DSets, tf_scope
from mx.visualizer import Visualization


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

    def load(self, batch_size, test_batch_size, force_cache_reload: bool = False):
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

        dsets = dsets.batch(batch_size, test_batch_size)

        return dsets

    def configure(self, task: Task):

        if isinstance(task, VectorSequenceMSE) or isinstance(task, RandomSequenceMSE):
            task.recieve_dataset_config(task.ds_config_cls(
                seq_dims=[28, 28],
                n_input_dims=1, # grayscale
            ))
        elif isinstance(task, RandomTokens):
            task.recieve_dataset_config(task.ds_config_cls(
                seq_dims=[28, 28],
                n_input_dims=1,
                codebook=tf.constant([0, 64, 85, 128, 169, 191, 233, 253], tf.uint8)[:, None], # add feat dim
            ))
        else:
            raise NotImplementedError(f"Dataset {type_name(self)} does not support Task {type_name(task)}. If using autoreload in IPython, try restarting the interpreter.")


    def get_visualizations(self, model, output_dir) -> dict[str, Visualization]:

        return u.list_to_box([
            MNISTImageViz(
                model,
                output_dir,
            )
        ])


class MNISTImageViz(Visualization):
    """
    Produce a HoloMap to display predicted MNIST images.
    """

    def __init__(
        self,
        model,
        name="mnist_images",
        desc="MNIST images",
        output_dir=None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            output_dir=output_dir,
        )

        self.model = model
        "Model to use for predictions"

    def __call__(self, timestep=None, pm: Progress=None, show=False) -> list:

        outs = predict_mnist_data(self.model, from_scratch=True, random_order=False, pm=pm)

        outs.numpy()

        n_seeds = outs.mean_viz_anims.shape[0]
        seed_dim = hv.Dimension(("seed", "Seed"))
        n_predict_steps = outs.mean_viz_anims.shape[1]
        predict_step_dim = hv.Dimension(("predict_step", "Predict Step"))
        if outs.samples_viz_anims is not None:
            n_samples = outs.samples_viz_anims.shape[1]
            samples_dim = [hv.Dimension(("sample", "Sample"))]
        else:
            n_samples = 1
            samples_dim = []

        def img(data, title):
            return hv.Image(data).opts(
                title=title,
                cmap="gray",
                aspect='equal',
            )

        def key(seed, sample, predict_step):

            if (sample is None or n_samples == 1) and predict_step is None:
                return (seed,)
            elif (sample is None or n_samples == 1):
                return (seed, predict_step)
            elif predict_step is None:
                return (seed, sample)
            else:
                return (seed, sample, predict_step)

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

        select_steps = list(u.expsteps(n_predict_steps))

        hmaps = []
        hmaps.append(hv.GridSpace(
            {
                key(i_seed, None, None): img(outs.seed_data_viz[i_seed], f"Seed data")
                for i_seed in range(n_seeds)
            },
            kdims=[predict_step_dim],
            label="Seed data",
        ))
        hmaps.append(hv.GridSpace(
            {
                key(i_seed, i_sample, None): img(outs.sampling_order_viz[i_seed, i_sample], f"Sampling order {i_sample}")
                for i_seed in range(n_seeds)
                for i_sample in range(n_samples)
            },
            kdims=[predict_step_dim, *samples_dim],
            label="Sampling order",
        ))
        hmaps.append(hv.GridSpace(
            {
                i_seed: hv.HoloMap({
                    i_predict_step: img(outs.mean_viz_anims[i_seed, i_predict_step], f"Mean Prediction")
                    for i_predict_step in select_steps
                }, kdims=[predict_step_dim])
                for i_seed in range(n_seeds)
            },
            kdims=[seed_dim],
            label="Mean Predictions",
        ))
        if outs.mean_entropy_anims is not None:
            hmaps.append(hv.GridSpace(
                {
                    i_seed: hv.HoloMap(
                        {
                            key(i_predict_step): img(outs.mean_viz_anims[i_seed, i_predict_step], f"Sample {i_seed}")
                            for i_predict_step in select_steps
                        },
                        kdims=[predict_step_dim]
                    )
                    for i_seed in range(n_seeds)
                },
                kdims=[seed_dim],
                label="Entropy of remaining locations (re. mean)",
            ))
        if outs.samples_viz_anims is not None:
            hmaps.append(hv.GridSpace(
                {
                    i_seed: hv.HoloMap({
                        (i_predict_step, i_sample): img(outs.mean_viz_anims[i_seed, i_sample, i_predict_step], f"Sample {i_seed}")
                        for i_predict_step in select_steps
                        for i_sample in range(n_samples)
                    }, kdims=[predict_step_dim, samples_dim])
                    for i_seed in range(n_seeds)
                },
                kdims=[seed_dim],
                label="Sampled Predictions",
            ))
        if outs.samples_viz_anims is not None:
            hmaps.append(hv.GridSpace(
                {
                    i_seed: hv.HoloMap({
                        (i_predict_step, i_sample): img(outs.mean_viz_anims[i_seed, i_sample, i_predict_step], f"Sample {i_seed}")
                        for i_predict_step in select_steps
                        for i_sample in range(n_samples)
                    }, kdims=[predict_step_dim, samples_dim])
                    for i_seed in range(n_seeds)
                },
                kdims=[seed_dim],
                label="Entropy of remaining locations (re. samples)",
            ))

        fig = hv.Layout(hmaps).cols(1).opts(
            shared_axes=False,
            title="MNIST Images",
            width=1600,
            sizing_mode="stretch_both",
            # responsive=True, # doesn't work
        )

        filename = "mnist"
        if timestep is not None:
            filename += f"_timestep{timestep}"
        if outs.seed_data_viz is None:
            filename += "_unseeded"
        filename += ".html"

        output_dir = Path(self.output_dir or "_mnist_plots")
        output_dir.mkdir(exist_ok=True, parents=True)

        filename = output_dir / filename

        hv.save(fig, filename, widget_location='top', fmt='html', backend='bokeh')

        if show:
            u.show(filename.absolute().as_uri(), title)



def predict_mnist_data(model, from_scratch=False, random_order=False, targeted=True, seed_len=784//2, ds: MxMNIST = None, pm: Progress=None) -> PredictOutputs:

    ds = ds or MxMNIST()

    inputs = next(iter(ds.load(10, 10).test))

    inputs = {
        "values": ein.rearrange(
            inputs["image"],
            "b h w c -> b (h w) c",
        ),
        "idxs": inputs["idxs"],
    }


    if random_order:
        idxs = tf.stack([
            tf.random.shuffle(
                u.multidim_indices([28, 28])
            )
            for _ in range(len(inputs["values"]))
        ], axis=0)
        values = tf.gather_nd(values, idxs, batch_dims=1)

    if model is None:
        seed_data = inputs["values"]
        idxs = inputs["idxs"]
        target_data = None
        target_data_idxs = None
    elif from_scratch:
        seed_data = None
        idxs = inputs["idxs"][:1]
        target_data = None
        target_data_idxs = None
    elif targeted:
        target_data = inputs["values"]
        target_data_idxs = inputs["idxs"]
        seed_data = tf.concat([
            inputs["values"][:, -1:],
            inputs["values"][:, :1],
        ], axis=1)
        idxs = tf.concat([
            inputs["idxs"][:, -1:],
            inputs["idxs"][:, :-1],
        ], axis=1)
    else:
        seed_data = inputs["values"][:, :seed_len]
        idxs = inputs["idxs"]
        target_data = inputs["values"]
        target_data_idxs = inputs["idxs"]

    if seed_data is not None and 'context/values' in model.input_shape:
        seed_data = tf.cast(seed_data, tf.float32) / 255.0
    elif seed_data is not None and 'context/tokens' in model.input_shape:
        seed_data = tf.cast(seed_data, tf.int32)

    outs = predict(
        name="Predict MNIST",
        desc=f"Predicting BVH data...",
        cfg=PredictInputs(
            model=model,
            out_seq_shape=[28, 28],
            raw_feat_shape=[1],
            viz_feat_shape=[3],
            target_data=target_data,
            target_data_idxs=target_data_idxs,
            seed_data=seed_data,
            idxs=idxs,
        ),
        pm=pm,
    )

    return outs


if __name__ == '__main__':
    u.set_debug()
    mnist = MxMNIST()
    batch_size = 10
    data = next(iter(mnist.load(batch_size, batch_size).test))
    tp(data, "MxMNIST data format")
    viz = MNISTImageViz(model=None)
    with create_progress_manager() as pm:
        viz(pm=pm)
