from box import Box

def get():
    return Box({
        "batch_size": 4,
        "shuffle_buffer_size": 1000,
        "force": False,
        "convert_deg_to_rad": True,
        "cached_dataset_path": "./cache/tf_dataset",
        "n_hands": 2,
        "n_dof": 23,
        # chunk size is used for models that take a fixed input size
        # it is a number of animation frames, and can be no smaller than
        # the shortest animation in the dataset
        "chunk_size": 100,
        # predict frames is the number of frames to predict when
        # vizualizing after each epoch
        "predict_frames": 30,
        "embd_dim": 64,
    })
