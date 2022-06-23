from box import Box

def get():
    return Box({
        "batch_size": 4,
        "shuffle_buffer_size": 130,
        "force": False,
        "convert_deg_to_rad": True,
        "cached_dataset_path": "./cache/tf_dataset",
        "n_hands": 1,
        "n_dof": 23,
        "decimated": True,
        "recluster": True,
        "vector": False,

        "target_is_sequence": True,
        "relative_frame_idxs": False,
        "scale_to_1_1": False,
        # Chunk size is used for models that take a fixed input size.
        # It is a number of animation frames, and can be no larger than
        # the shortest animation in the dataset
        "chunk_size": 2,
        # predict frames is the number of frames to predict when
        # vizualizing after each epoch
        "predict_frames": 20,
        "n_test_samples": 3,

        "learning_rate": 0.0005,

        "warmup_steps": 3000,

        # Model config
        "embd_dim": 512,
        "dropout_rate": 0.1,


        "multi_lstm": {
            "n_layers": 10,
        },
        "mlp": {
            "layers": [1024],
        },
        "conv": {
            "filters": 16,
            "width_frames": 1,
        },
        "transformer": {
            "initializer_range": 0.01,
            "n_enc_layers": 1,
            "ffl_dim": 1024,
            "n_heads": 4,
            "activation": "relu",
        }
    })
