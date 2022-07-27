from box import Box

def get():
    return Box({
        "shuffle_buffer_size": 130,
        "force": False,
        "convert_deg_to_rad": True,
        "cached_dataset_path": "./cache/tf_dataset",
        "n_hands": 2,
        "n_dof": 23,
        "decimate": True,
        "decimate_threshold": 0.3,
        "recluster": True,
        "vector": False,

        "target_is_sequence": True,
        "relative_frame_idxs": False,
        "scale_to_1_1": False,
        # Chunk size is used for models that take a fixed input size.
        # It is a number of animation frames, and can be no larger than
        # the shortest animation in the dataset
        "chunk_size": 6,
        # predict frames is the number of frames to predict when
        # vizualizing after each epoch
        "predict_frames": 20,
        "test_batch_size": 3,

        "batch_size": 16,
        "steps": 100000,
        "steps_per_epoch": 1000,

        "optimizer": "warmup_sgd",
        "adam": {
            "lr": 0.0005,
        },
        "warmup_sgd": {
            "momentum": 0.9,
            "clip_norm": 1.0,
            "lr": 0.01,
            "warmup_steps": 10000,
        },

        # Model config
        "embd_dim": 512,

        "mlp": {
            "hidden_dim": 1024,
            "n_layers": 1,
            "activation": "relu",
            "dropout_rate": 0.1,
        },
        "conv": {
            "filters": 16,
            "width_frames": 1,
            "activation": "relu",
            "dropout_rate": 0.1,
        },
        "transformer": {
            "initializer_range": 0.01,
            "n_layers": 6,
            "ffl_dim": 4096,
            "n_heads": 8,
            "activation": "gelu",
            "dropout_rate": 0.1,
        },
        "deberta": {
            "num_attention_heads": 4,
            "hidden_size": 256,
            "intermediate_size": 1024,
            "initializer_range": 0.01,
            "num_hidden_layers": 3,
            "hidden_act": "gelu",
            "relative_attention": True,
            "max_relative_positions": 50, # maximum number of frames in context window. Determines the size of the relative position matrix.
            "pos_att_type": ["c2c", "c2p", "p2c"],
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "layer_norm_eps": 1e-07,
        }
    })
