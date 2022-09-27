import tensorflow_datasets as tfds
import tensorflow as tf
import typing
if typing.TYPE_CHECKING:
    import keras.api._v2.keras as keras
    from keras.api._v2.keras import Model, Input, layers
else:
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers

from mx import bvh, datasets, train, layers, utils


seq_len = 32

dataset, shapes = datasets.init_dataset_and_task(
    datasets.bvh.BvhAllColumns(
        recluster=True,
        decimate=0.5,
    ),
    datasets.tasks.NextVectorPrediction(
        batch=8,
        sequence_length=seq_len,
    ),
)

n_layers = 5
embd_dim = 256

blocks = [
    l for i in range(n_layers)
    for l in [
        layers.mha(
            n_heads=8,
            weight_type="softmax",
            seq_dim=seq_len,
            embd_dim=embd_dim,
            name=f"mha_{i}"
        ),
        layers.mlp(
            embd_dim=embd_dim,
            hidden_units=4096,
            name=f"mlp_{i}"
        ),
    ]
]

backbone = layers.residual(
    embd_dim=embd_dim,
    layers=blocks,
    seq_dims=[seq_len],
    name="backbone"
)

embedding = layers.codebook(
    num_tokens=512,
    embd_dim=embd_dim,
    seq_dims=[seq_len],
    name="codebook"
)

head = layers.circular_mse(
    in_dims=shapes["inputs"]["input"].with_feature_dims({ "embd": embd_dim }),
)

inp = Input(shape=[seq_len, 3], name="inp")

model = Model(inputs=inp, outputs=head(backbone(embedding(inp))), name="model")
