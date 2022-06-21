from box import Box

def get():
    return Box({
        "batch_size": 4,
        "shuffle_buffer_size": 1000,
        "force": False,
        "convert_deg_to_rad": True,
        "cached_dataset_path": "./cache/tf_dataset",
        "n_hands": 1,
        "n_dof": 6,

        "target_is_sequence": False,
        "relative_frame_idxs": True,
        # Chunk size is used for models that take a fixed input size.
        # It is a number of animation frames, and can be no larger than
        # the shortest animation in the dataset
        "chunk_size": 40,
        # predict frames is the number of frames to predict when
        # vizualizing after each epoch
        "predict_frames": 20,

        "learning_rate": 0.0005,

        "warmup_steps": 3000,

        # Model config
        "embd_dim": 256,
        "dropout_rate": 0.1,

        "multi_lstm": {
            "n_layers": 10,
        },
        "mlp": {
            "layers": [1024],
        },
        "transformer": {
            "n_enc_layers": 2,
            "ffl_dim": 515,
            "n_heads": 8,
            "activation": "relu",
        }
    })
