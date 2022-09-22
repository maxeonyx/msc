from dataclasses import dataclass
from typing_extensions import Self

import tensorflow as tf

@dataclass
class DSet:
    train: tf.data.Dataset
    test: tf.data.Dataset
    val: tf.data.Dataset

    def destructure(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        return self.train, self.test, self.val

    def map(self, fn) -> Self:
        return DSet(
            train = self.train.map(fn),
            test = self.test.map(fn),
            val = self.val.map(fn),
        )
