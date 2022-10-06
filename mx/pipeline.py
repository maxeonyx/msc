from dataclasses import dataclass

from mx.datasets import Dataset, DSets
from mx.tasks import Task
from mx.embedding import Embedding
from mx.models import MxModel

from mx.utils.tf import *

@dataclass
class Pipeline:

    def __init__(self, dataset: Dataset, task: Task, embedding: Embedding, model: MxModel) -> None:

        if not dataset.compatible_with(task):
            raise NotImplementedError(f"{type(dataset)} cannot be used for {type(task)}")
        if not task.compatible_with(embedding):
            raise NotImplementedError(f"{type(task)} cannot be used for {type(embedding)}")
        if not embedding.compatible_with(model):
            raise NotImplementedError(f"{type(embedding)} cannot be used for {type(model)}")

        self.dataset = dataset
        self.task = task
        self.embedding = embedding
        self.model = model
    
    def _load(self) -> DSets:

        data = self.dataset.load_and_adapt_for(self.task)

        input_data = self.task.adapt_for(self.embedding, data)

        return input_data
    
    def get_train_data(self, batch_size: int, n_steps: int) -> tf.data.Dataset:

        dsets = self._load()

        ds_train = dsets.train
        
        if not self.task.does_batching:
            ds_train = ds_train.batch(batch_size)
        
        ds_train = (
            ds_train
            .take(n_steps)
            .enumerate()
            .map(lambda i, x: (i, (x["inputs"], x["targets"])))
        )

        return ds_train


    
