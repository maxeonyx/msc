from typing import TypedDict

from mx.utils import Einshape

from ._dataset_utils import *
from . import tasks
from . import bvh

class DatasetShape(TypedDict):
    inputs: dict[str, Einshape]
    targets: dict[str, Einshape]
    extra: dict[str, Einshape]

def init_dataset_and_task(d_cfg: BaseDatasetConfig, t_cfg: tasks.TaskConfig) -> tuple[DSet, DatasetShape]:

    if isinstance(d_cfg, bvh.BvhAllColumns):

        if isinstance(t_cfg, tasks.NextVectorPrediction):
            return bvh.vector_ntp()
