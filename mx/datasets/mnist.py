from mx.prelude import *

from mx.pipeline import MxDataset, Task
from mx.visualizer import Visualization


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
        train_val_split=(5/6, 1/6),
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

    def load(self, force_reload=False):
        """
        Load the dataset from disk.
        """
        mnist = tfds.load("mnist", shuffle_files=True)

        tp(mnist)

        tp(next(iter(mnist["train"])))

    def configure(self, task: Task):
        pass

    def get_visualizations(self, viz_batch_size, task_specific_predict_fn) -> dict[str, Visualization]:
        return {}


if __name__ == '__main__':
    u.set_debug()
    mnist = MxMNIST()
    mnist.load()
