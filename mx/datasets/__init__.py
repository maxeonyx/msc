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
            task, shapes = bvh.vector_ntp(data_cfg, task_cfg, force_cache_reload=force_cache_reload)


    # setup task for train loop
    # - batch
    # - window for fused steps
    # - window for epochs
    ds_test, ds_val, ds_train = task.destructure()
    if not task_cfg.already_batched():
        ds_train = ds_train.batch(train_cfg.batch_size)
        ds_val = ds_val.batch(train_cfg.test_batch_size)
        ds_test = ds_test.batch(train_cfg.test_batch_size)

        train_shapes = DatasetShape(
            inputs={ k: shp.batch(train_cfg.batch_size) for k, shp in shapes.inputs.items() },
            targets={ k: shp.batch(train_cfg.batch_size) for k, shp in shapes.targets.items() },
            extra={ k: shp.batch(train_cfg.batch_size) for k, shp in shapes.extra.items() },
        )

        test_val_shapes = DatasetShape(
            inputs={ k: shp.batch(train_cfg.test_batch_size) for k, shp in shapes.inputs.items() },
            targets={ k: shp.batch(train_cfg.test_batch_size) for k, shp in shapes.targets.items() },
            extra={ k: shp.batch(train_cfg.test_batch_size) for k, shp in shapes.extra.items() },
        )

        shapes = DataPipelineShape(
            train=train_shapes,
            test=test_val_shapes,
            val=test_val_shapes,
        )

    ds_train = ds_train.take(train_cfg.n_steps).enumerate()
    ds_train = ds_train.window(train_cfg.fused_steps).enumerate()
    ds_train = ds_train.window(train_cfg.n_steps_per_epoch//train_cfg.fused_steps).enumerate()
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    data_pipeline = DSet(
        train=ds_train,
        test=ds_test,
        val=ds_val,
    )

    return data_pipeline, shapes
