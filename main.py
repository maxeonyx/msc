import os
from pathlib import Path

try:
    run_name = os.environ["RUN_NAME"]
except KeyError:
    import randomname
    run_name = randomname.get_name()
    os.environ["RUN_NAME"] = run_name

from mx import progress
with progress.create_progress_manager(run_name) as pm:

    with pm.enter_spinner("Init Tensorflow", "Initializing Tensorflow..."):
        from mx.prelude import *
        from mx import train
        from mx.datasets.bvh import BvhDataset
        from mx.tasks import NextUnitVectorPrediction
        from mx.embedding import TransformerAngleVectorEmbedding
        from mx.models import DecoderOnlyTransformer
        from mx.pipeline import Pipeline

    with pm.enter_spinner("Init Datasets", "Initializing Datasets..."):

        # Decoder-only transformer for BVH data vector regression
        pipeline = Pipeline(
            dataset=BvhDataset(
                recluster=False,
                decimate=False,
            ),
            task=NextUnitVectorPrediction(
                chunk_size=83,
                pred_seed_len=5,
            ),
            embedding=TransformerAngleVectorEmbedding(
                n_embd=14,
                n_repeats=11,
            ),
            model=DecoderOnlyTransformer(
                n_layers=2,
                n_heads=13,
                n_hidden=67,
            ),
            # bvh-angle-vector-regression-decoder-only
            # bvh-ang-vec-reg-dec-only
            # b-a-vec-dec-o
            # b-avec-do
            # bavecdo
            identifier="bavecdo",
        )

        output_dir = Path("_outputs") / pipeline.identifier / run_name

        model = pipeline.get_model()
        loss_fn = pipeline.get_loss_fn()
        data = pipeline.get_train_data(
            batch_size=31,
            n_steps=5000,
        )
        vizr = pipeline.get_visualizer(
            output_dir=output_dir / "viz",
            viz_batch_size = 3,
            cfgs = {
                "bvh_imgs": {
                    "render_on": ["start", "epoch"],
                    "show_on": ["start", "end", "exit"]
                },
            },
        )

    train.train_loop(
        pm,
        data,
        model,
        loss_fn,
        pipeline.identifier,
        run_name,
        vizr,
        profile=False,
        compile=True,
    )
