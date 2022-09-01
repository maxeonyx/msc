from dataclasses import dataclass
from box import Box as box

def get():

    irmqa_cfg = box({
        "qk_dim": 19, # 16w * 8h = 128 ~= 102
        "v_dim": 102,
        "n_heads": 7,
        "intermediate_size": 1048,
        "initializer_range": 0.02,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "layer_norm_eps": 1e-7,
    })



    return box({
        
        "shuffle_buffer_size": 130,
        "force": False,
        "convert_deg_to_rad": True,
        "cached_dataset_path": "./cache/tf_dataset",

        "decimate": True,
        "decimate_threshold": 0.3,
        "recluster": True,

        "pre": {
            "scale_to_1_1": False,
            # Chunk size is used for models that take a fixed input size.
            # It is a number of animation frames, and can be no larger than
            # the shortest animation in the dataset
            "chunk_size": 6,
            "n_hands": 2,
            "n_dof": 23,
            "columns": "useful",
        },
    
        "dream": {
            "n_hand_vecs": 18,
            "batch_size": 11,

            "decimate": False,
            "decimate_threshold": 0.3,
            "recluster": False,

            "l2_reg": 1e-6,
            "l1_reg": 0,


            "embd_dim": 1 * 1 * 3 * 5 * 2 * 10,

            "task": "flat",

            "ds": "real",

            "ds_synthetic": {
                "n_hands": 1,
                "n_joints_per_hand": 1,
                "n_dof_per_joint": 3,

                # num sin components
                "n_sins": 3,
            },

            "ds_real": {
                "n_hands": 1,
                "n_joints_per_hand": 1,
                "n_dof_per_joint": 3,

                "columns": "all",
            },

            "task_flat": {
                "n_examples": 60,
                "n_hand_vecs": 200,
                "random_ahead": False,

                "model": {
                    "max_rel_embd": 1000,
                    "n_rotations": 5,
                    "decoder": {
                        "n_layers": 3,
                        **irmqa_cfg,
                        "hidden_act": "gelu",
                    },
                },
            },

            "task_flat_query": {
                "n_examples": 60,
                "n_hand_vecs": 200,
                "random_ahead": False,

                "model": {
                    "max_rel_embd": 1000,
                    "n_rotations": 5,
                    "encoder": {
                        "n_layers": 3,
                        **irmqa_cfg,
                        "hidden_act": "gelu",
                    },
                    "decoder": {
                        "n_layers": 1,
                        **irmqa_cfg,
                        "hidden_act": "gelu",
                    },
                },
            },

            "task_hierarchical": {
                "contiguous": True,

                "model": {
                    "max_rel_embd": 1000,
                    "hand_encoder": {
                        "n_layers": 3,
                        **irmqa_cfg,
                    },
                    "hand_decoder": {
                        **irmqa_cfg,
                    },
                    "joint_decoder": {
                        **irmqa_cfg,
                    },
                    "dof_decoder": {
                        **irmqa_cfg,
                    },
                },
            },
        },

        "decoder_only": {
            "n_hands": 1,
            "n_dof": 3,
            "n_joints_per_hand": 1,
            "n_dof_per_joint": 3,
            "columns": "all",
            "n_hand_vecs": 18,
            "batch_size": 11,
        },

        # predict frames is the number of frames to predict when
        # vizualizing after each epoch
        "predict_frames": 20,
        "test_batch_size": 3,

        "steps": 100000,
        "test_steps": 1000,
        "val_steps": 1000,
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
