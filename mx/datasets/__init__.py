from pprint import pprint
from re import I

from mx.utils.tf import *
from ._dataset_utils import *
from . import tasks
from . import bvh
from mx import train

def init_data_pipeline(
    data_cfg: BaseDatasetConfig,
    task_cfg: tasks.TaskCfg,
    train_cfg: tasks.TrainingCfg,
    force_cache_reload: bool = False,
) -> tuple[DSet, DataPipelineShape]:

    # create task
    if isinstance(data_cfg, bvh.BvhAllColumns):

        if isinstance(task_cfg, tasks.NextVectorPrediction):
            dset, shapes = bvh.vector_ntp(data_cfg, task_cfg, force_cache_reload=force_cache_reload)

    # count_calls_fn, count_var = count_calls()
    # dset.train = dset.train.map(count_calls_fn)

    # setup task for train loop
    # - batch
    # - window for fused steps
    # - window for epochs
    if not task_cfg.already_batched():
        dset, shapes = dset.batch(train_cfg.batch_size, train_cfg.test_batch_size, shapes)

    # count_calls_fn_2, count_var_2 = count_calls()
    # dset.train = dset.train.map(count_calls_fn_2)

    def window_nested(ds, window_size):
        """
        Work around the frustrating behaviour of tf.data.Dataset.window() which
        windows from "the inside out" rather than "the outside in", when applied to
        nested datasets.
        """
        ds = ds.window(window_size)
        ds = ds.map(lambda *x: tf.data.Dataset.zip(x))
        return ds
    
    # dset = dset.map(inspect("pre_enumerate"))

    n_fusedsteps_per_epoch = train_cfg.n_steps_per_epoch // train_cfg.fused_steps
    dset.train = dset.train.take(8000).enumerate() # steps
    dset.train = window_nested(dset.train, train_cfg.fused_steps).enumerate() # fused steps
    dset.train = window_nested(dset.train, n_fusedsteps_per_epoch).enumerate() # epochs
    
    # n_epochs = dset.train.cardinality()
    # print(f"n_epochs: {n_epochs}")
    # for i_epoch, epoch in dset.train:
    #     print(f"epoch {i_epoch} / {n_epochs}")
    #     n_fusedsteps = epoch.cardinality()
    #     print(f"n_fusedsteps: {n_fusedsteps}")
    #     for i_fusedstep, fusedstep in epoch:
    #         print(f"  fusedstep {i_fusedstep}")
    #         n_steps = fusedstep.cardinality()
    #         print(f"  n_steps: {n_steps}")
    #         for i_step, step in fusedstep:
    #             print(f"    step {i_step}")
    
    # print(f"count (pre batch): {count_var.numpy()}")
    # print(f"count (post batch): {count_var_2.numpy()}")
    
    # raise Exception("stop")

    # data_pipeline = DSet(
    #     train=ds_train,
    #     test=ds_test,
    #     val=ds_val,
    # )

    return dset, shapes
