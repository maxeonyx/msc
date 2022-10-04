from pprint import pprint
from re import I
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


    # setup task for train loop
    # - batch
    # - window for fused steps
    # - window for epochs
    if not task_cfg.already_batched():
        dset, shapes = dset.batch(train_cfg.batch_size, train_cfg.test_batch_size, shapes)
        
    
    print("ds train cardinality (after batch)", dset.train.cardinality().numpy())
    print("ds test cardinality (after batch)", dset.test.cardinality().numpy())
    print("ds val cardinality (after batch)", dset.val.cardinality().numpy())

    def nonfucked_window(ds, window_length):
        """
        Work around the frustrating behaviour of .window() where it applies the
        windowing from the inside out instead of the outside in.
        """
        dd = ds.window(window_length).enumerate()
        dd = dd.map(lambda i, x: (i, tf.data.Dataset.zip(x)))
        return dd
    
    
    dset.train = dset.train.take(train_cfg.n_steps).enumerate()
    
    print("ds train cardinality (after take)", dset.train.cardinality().numpy())
    print("ds test cardinality (after take)", dset.test.cardinality().numpy())
    print("ds val cardinality (after take)", dset.val.cardinality().numpy())

    dset.train = dset.train.window(train_cfg.n_steps).enumerate()

    print("ds train cardinality (after window)", dset.train.cardinality().numpy())
    print("ds test cardinality (after window)", dset.test.cardinality().numpy())
    print("ds val cardinality (after window)", dset.val.cardinality().numpy())

    fusedsteps_per_epoch = train_cfg.n_steps_per_epoch // train_cfg.fused_steps
    dset.train = dset.train.window(fusedsteps_per_epoch).enumerate()
    
    print("ds train cardinality (after window 2)", dset.train.cardinality().numpy())
    print("ds test cardinality (after window 2)", dset.test.cardinality().numpy())
    print("ds val cardinality (after window 2)", dset.val.cardinality().numpy())

    zzip = tf.data.Dataset.zip

    # dset.train = dset.train.prefetch(tf.data.experimental.AUTOTUNE)
    for i_e, epoch in dset.train.take(1):
        print(f"epoch {i_e.numpy()}")
        for i_fs, fusedstep in zzip(epoch).take(1):
            print(f"  fusedstep {i_fs.numpy()}")
            for i_s, val in zzip(fusedstep).take(1):
                print(f"    step {i_s.numpy()}")
                break
                # pprint(depth=4, indent=2, object=tf.nest.map_structure(lambda x: tf.shape(x), val))
    print("HIII")
    raise Exception("stop")

    data_pipeline = DSet(
        train=ds_train,
        test=ds_test,
        val=ds_val,
    )

    return data_pipeline, shapes
