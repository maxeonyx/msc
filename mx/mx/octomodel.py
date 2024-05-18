from mx.predict import predict_core
from mx.prelude import *
from mx import datasets as mxd, tasks as mxt, embeddings as mxe, models as mxm


# @export
# class MxModel(tf.Module):
#     """
#     Wrapper for keras model that just adds a desc attribute.
#     """

#     def __init__(self, *args, desc: str, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.desc = desc


@export
class MNISTPreprocess(u.MxModule):

    def __init__(
        self,
        chunk_size: int,
        task: Literal['next', 'rand'],
        val_embd: Literal['code', 'scalar'],
        **kwargs,
    ):
        super().__init__(
            name="mnist_preprocess",
            desc=f"MNISTPreprocess(chunk_size={chunk_size}, task={task}, val_embd={val_embd})",
            **kwargs,
        )
        self.chunk_size = chunk_size
        self.task = task
        self.val_embd = val_embd

    def build(
        self,
        input_spec,
    ):
        self.image_to_seq = layers.Lambda(lambda x: {
            'vals': ein.rearrange(
                x['image'],
                'b h w c -> b (h w) c',
            ),
            'idxs': x['idxs'],
        })
        if self.task == "next":
            self.chunk = mxt.RandomChunk(self.chunk_size)
        elif self.task == "rand":
            self.chunk = mxt.RandomSlices(self.chunk_size)
        else:
            raise NotImplementedError(f"Task {self.task} not implemented for MNIST")

        if self.val_embd == "scalar":
            self.convert_in = layers.Lambda(lambda x: {
                'vals': tf.cast(x['vals'], u.dtype()) / 255.,
                'idxs': x['idxs'],
            })
        elif self.val_embd == "code":
            codebook = mxd.mnist.mnist_codebook()
            self.discretize = mxt.Discretize(codebook)

        self.to_model_format = layers.Lambda(lambda x: {
            'ctx_inp_vals': x['vals'][:, :-1],
            'ctx_inp_idxs': x['idxs'][:, :-1],
            'ctx_tar_idxs': x['idxs'],
        })

        self.to_train_format = layers.Lambda(lambda x: (
            self.to_model_format(x), # input
            x['vals'], # target
        ))

    @tf.function
    def image_to_seq(self, image):
        return {
            'vals': ein.rearrange(
                image,
                'b h w c -> b (h w) c',
            ),
            'idxs': tf.range(image.shape[1] * image.shape[2]),
        }

    @tf.function
    def to_model_format(self, x):
        return {
            'ctx_inp_vals': x['vals'][:, :-1],
            'ctx_inp_idxs': x['idxs'][:, :-1],
            'ctx_tar_idxs': x['idxs'],
        }


@export
class MNISTModel(u.MxModule):
    """
    Highly-configurable model for MNIST tasks.

    >>> dataset = mxd.MxMNIST()
    >>> data = next(iter(dataset.load(128, 128).test))

    >>> model = MNISTModel(task='next')
    >>> inputs, targets = model.preprocess(data)
    >>> model(inputs).shape
    TensorShape([128, 32, 1])

    >>> model = MNISTModel(task='rand')
    >>> inputs, targets = model.preprocess(data)
    >>> model(inputs).shape
    TensorShape([128, 32, 1])

    >>> model = MNISTModel(val_embd='code')
    >>> inputs, targets = model.preprocess(data)
    >>> inputs['ctx_inp_vals'].dtype
    tf.int64
    >>> model(inputs).shape
    TensorShape([128, 32, 1])

    >>> model.build({ 'image': tf.TensorShape([None, 28, 28, 1]), 'idxs': tf.TensorShape([None, 28, 28, 2]), 'label': tf.TensorShape([None]) })
    >>> dir = Path("/tmp/mnistmodel")
    >>> dir.mkdir(exist_ok=True)
    >>> tf.saved_model.save(model, dir)
    """

    def __init__(
        self,
        chunk_size=32,
        task='next',
        val_embd='scalar',
        pos_embd='code',
        n_embd=256,
        model_cfg=Box(
            type='transformer',
            n_layers=6,
            n_hidden=512,
            n_heads=8,
            dropout=0.5,
            use_layernorm=True,
            use_learned_add=True,
            use_batchnorm=True,
        ),
        dist_cfg=None,
        name="mnistmodel",
    ):
        super().__init__(name=name, desc=f"MNISTModel({model_cfg.type})")

        self.chunk_size = chunk_size
        self.task = task
        self.val_embd = val_embd
        self.pos_embd = pos_embd
        self.n_embd = n_embd
        self.model_cfg = model_cfg
        self.dist_cfg = dist_cfg

    def build(self, input_shape):
        assert 'image' in input_shape, f"Expected 'vals' in input_shape, got {input_shape}"
        assert 'idxs' in input_shape, f"Expected 'idxs' in input_shape, got {input_shape}"
        assert 'label' in input_shape, f"Expected 'label' in input_shape, got {input_shape}"

        self.pre = Box()



        if self.val_embd == "scalar":
            val_embedder = mxe.ScalarEmbd(self.n_embd)
        elif self.val_embd == "code":
            val_embedder = mxe.CodebookEmbd(self.n_embd, len(codebook))

        if self.pos_embd == "code":
            inp_pos_embedders = [
                mxe.CodebookEmbd(self.n_embd, self.chunk_size, name=f"pos_embd_{dim}")
                for dim in ["h", "w"]
            ]
            if self.task in ["rand", "targ"]:
                tar_pos_embedders = [
                    mxe.CodebookEmbd(self.n_embd, self.chunk_size, name=f"pos_embd_{dim}")
                    for dim in ["h", "w"]
                ]
            else:
                tar_pos_embedders = None

        self.embedder = mxe.DecoderOnlyEmbedding(
            val_embedder=val_embedder,
            inp_pos_embedders=inp_pos_embedders,
            tar_pos_embedders=tar_pos_embedders,
        )

        if self.model_cfg.type == 'transformer':
            self.backbone = mxm.DecoderOnlyTransformer(
                n_layers=self.model_cfg.n_layers,
                n_heads=self.model_cfg.n_heads,
                n_hidden=self.model_cfg.n_hidden,
                dropout=self.model_cfg.dropout,
                use_batchnorm=self.model_cfg.use_batchnorm,
                use_layernorm=self.model_cfg.use_layernorm,
                use_learned_add=self.model_cfg.use_learned_add,
            )
        elif self.model_cfg.type == 'resnet':
            self.backbone = mxm.Resnet(
                n_layers=self.model_cfg.n_layers,
                n_hidden=self.model_cfg.n_hidden,
                dropout=self.model_cfg.dropout,
                use_batchnorm=self.model_cfg.use_batchnorm,
                use_learned_add=self.model_cfg.use_learned_add,
            )
        elif self.model_cfg.type == 'mlp':
            self.backbone = mxm.LittleMLP(
                n_hidden=self.model_cfg.n_hidden,
                dropout=self.model_cfg.dropout,
            )
        else:
            raise NotImplementedError(f"Model {self.model_cfg.type} not implemented for MNIST")

        if self.dist_cfg is None:
            self.head = layers.Dense(1)
        else:
            raise NotImplementedError(f"Dist {self.dist_cfg.type} not implemented for MNIST")

        self.post = Box()

        if self.dist_cfg is None:
            # scale x from [-1, 1] to [0, 255], clip and cast to uint8
            self.post.convert_out = layers.Lambda(lambda x: tf.cast(tf.clip_by_value(
                (x + 1.) / 2. * 255.,
                0,
                255,
            ), tf.uint8))
        else:
            raise NotImplementedError(f"Dist {self.dist_cfg.type} not implemented for MNIST")

        self.built = True

    @tf.function(
        input_signature=[
            {
                'image': tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.uint8),
                'idxs': tf.TensorSpec(shape=(None, None, 2), dtype=tf.int32),
                'label': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            },
        ],
    )
    def preprocess(self, x):
        if not self.built:
            self.build(x)
        x = self.pre.image_to_seq(x)
        x = self.pre.chunk(x)
        if self.val_embd == "code":
            x = self.pre.discretize(x)
        else:
            x = self.pre.convert_in(x)
        x = self.pre.to_train_format(x)
        return x

    @tf.function(
        input_signature=[
            {

                'ctx_inp_vals': tf.TensorSpec(shape=(None, None), dtype=u.dtype()),
                'ctx_inp_idxs': tf.TensorSpec(shape=(None, None, 2), dtype=tf.int32),
                'ctx_tar_idxs': tf.TensorSpec(shape=(None, None, 2), dtype=tf.int32),
            },
        ],
    )
    def call(self, x):
        embd = self.embedder(x)
        embd = self.backbone(embd)
        out = self.head(embd)
        if self.dist_cfg is None:
            return out
        else:
            raise NotImplementedError(f"Dist {self.dist_cfg.type} not implemented for MNIST")

    def predict_core(
        model,
        i_step,
        start_at,
        break_var,
        sample_fns,
        viz_fns,
        vals_var: tf.Variable,
        idxs_var: tf.Variable,
        viz_var: tf.Variable,
        query_all: bool,
        sampling_order: Literal["fixed", "highest_entropy", "lowest_entropy"],
    ):
        predict_core(
            model,
            i_step,
            start_at,
            break_var,
            sample_fns,
            viz_fns,
            vals_var,
            idxs_var,
            viz_var,
            query_all,
            sampling_order,
        )



if __name__ == '__main__':
    import doctest
    doctest.testmod()

    model = MNISTModel()
