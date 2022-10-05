import os

from mx.datasets import tasks
try:
    run_name = os.environ["RUN_NAME"]
except KeyError:
    import randomname
    run_name = randomname.get_name()
    os.environ["RUN_NAME"] = run_name



from mx import progress
with progress.create_progress_manager(run_name) as pm:

    

    with pm.enter_spinner("Init Tensorflow", "Initializing Tensorflow..."):
        from mx.utils.tf import *
        from mx import datasets, layers, train

    with pm.enter_spinner("Init Dataset", "Initializing dataset and task..."):
        seq_len = 32
        dataset_config = datasets.bvh.BvhAllColumns(
            recluster=True,
            decimate=0.5,
        )
        task_config = datasets.tasks.NextVectorPrediction(
            sequence_length=seq_len,
        )
        train_cfg = tasks.TrainingCfg(
            batch_size=64,
            n_steps=5000,
            n_steps_per_epoch=500,
        )
        dataset, shapes = datasets.init_data_pipeline(dataset_config, task_config, train_cfg)
        inp_shape = shapes.train.inputs["input"]
        inp_idxs_shape = shapes.train.inputs["input_idxs"]
        tar_idxs_shape = shapes.train.inputs["target_idxs"]


    with pm.enter_spinner("Init Model", "Creating the model..."):
        n_layers = 5
        embd_dim = 256

        inp_embd_shape = inp_shape.with_feature_dims({ "embd": embd_dim })
        inp_embedding = layers.angle_embedding(
            num_repeats=5,
            input_shape=inp_shape,
            embd_shape=inp_embd_shape,
            name="inp_embd",
        )
        inp_idx_embedding = layers.codebook(
            n_tokens=512,
            embd_shape=inp_embd_shape,
            add_begin_token=True,
            name="inp_idx_embd"
        )
        prepend_begin_token = layers.prepend_begin_token(
            input_shape=inp_embd_shape,
            name="prepend_begin_token",
        )
        tar_idx_embedding = layers.codebook(
            n_tokens=512,
            embd_shape=tar_idxs_shape.with_feature_dims({ "embd": embd_dim }),
            add_begin_token=False,
            name="tar_idx_embd"
        )

        embd_shape = (
            inp_shape
            .with_sequence_dims({ "seq": seq_len })
            .with_feature_dims({ "embd": embd_dim })
        )

        blocks = [
            l for i in range(n_layers)
            for l in [
                layers.mha(
                    embd_shape=embd_shape,
                    n_heads=8,
                    name=f"mha_{i}"
                ),
                layers.featurewise_dense_block(
                    hidden_size=1024,
                    in_dims=embd_shape,
                    out_dims=embd_shape,
                    name=f"mlp_{i}"
                ),
            ]
        ]
        
        backbone = layers.residual(
            embd_shape=embd_shape,
            layers=blocks,
            name="backbone"
        )

        head = layers.circular_mse(
            target_dims=shapes.train.targets,
            embd_dims=embd_shape,
        )

        def call(input, input_idxs, target_idxs):
            input_embd = inp_embedding(input)
            input_idxs_embd = inp_idx_embedding(input_idxs)
            input_embd = input_embd + input_idxs_embd
            input_embd = prepend_begin_token(input_embd)
            target_idxs_embd = tar_idx_embedding(target_idxs)
            embd = input_embd + target_idxs_embd
            embd = backbone(embd)
            return head.final_layer(embd)

        inputs = layers.input_dict(
            Input(shape=inp_shape.s_f_shape, name="input"),
            Input(shape=inp_idxs_shape.s_f_shape, name="input_idxs"),
            Input(shape=tar_idxs_shape.s_f_shape, name="target_idxs"),
        )

        print(inputs)
        
        model = Model(inputs=inputs, outputs=call(**inputs), name="model")

    with pm.enter_spinner("Init Train Loop", "Building the training loop..."):
        train_loop = train.make_train_loop(
            train_cfg=train.TrainLoopCfg(),
            task_cfg=task_config,
            run_name=None, # random
            task=dataset,
            loss_fn=head.loss_fn,
            model=model,
        )

    train_loop(pm)
