import tensorflow_datasets as tfds
import tensorflow as tf
import typing
if typing.TYPE_CHECKING:
    import keras.api._v2.keras as keras
    from keras.api._v2.keras import Model, Input, layers
else:
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers

from mx import bvh, train, layers, utils
from mx.utils import Einshape

print(bvh.DEFAULT_BVH_DIR)
print(bvh.DEFAULT_OUTPUT_BVH_DIR)

n_layers = 5
embd_dim = 256
seq_len = 128

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
    seq_dim=[seq_len],
    name="codebook"
)

head = layers.circular_mse(
    embd_shape=Einshape(sequence_dims={"frames": seq_len}, embd_dims={"embd": embd_dim}),
)

inp = Input(shape=[seq_len, 3], name="inp")
