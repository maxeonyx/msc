import os

try:
    run_name = os.environ["RUN_NAME"]
except KeyError:
    import randomname
    run_name = randomname.get_name()
    os.environ["RUN_NAME"] = run_name



from mx import progress
with progress.create_progress_manager(run_name) as pm:

    

    with pm.enter_spinner("Init Tensorflow", "Initializing Tensorflow..."):
        from mx.tf import *
    
    with pm.enter_spinner("Init Datasets", "Initializing Datasets..."):
        from mx.datasets.bvh import BvhDataset
        from mx.tasks import NextUnitVectorPrediction
        from mx.embedding import TransformerAngleVectorEmbedding
        from mx.models import DecoderOnlyTransformer
        from mx.pipeline import Pipeline

        # Decoder-only transformer for BVH data vector regression
        pipeline = Pipeline(
            dataset=BvhDataset(
                recluster=False,
                decimate=False,
            ),
            task=NextUnitVectorPrediction(
                chunk_size=24,
            ),
            embedding=TransformerAngleVectorEmbedding(
                n_embd=256,
                n_repeats=7,
            ),
            model=DecoderOnlyTransformer(
                n_layers=3,
                n_heads=8,
                n_hidden=1024,
            ),
            # bvh-angle-vector-regression-decoder-only
            # bvh-ang-vec-reg-dec-only
            # b-a-vec-dec-o
            # b-avec-do
            # bavecdo
            identifier="bavecdo",
        )

        model = pipeline.get_model()
        loss_fn = pipeline.get_loss_fn()
        data = pipeline.get_train_data(
            batch_size=32,
            n_steps=5000,
        )

    from mx import train
    train.train_loop(
        pm,
        data,
        model,
        loss_fn,
        pipeline.identifier,
        run_name,
        n_steps_per_epoch=500,
    )
