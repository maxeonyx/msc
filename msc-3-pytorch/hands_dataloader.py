import torch
from torch.utils.data import DataLoader, RandomSampler
from random import shuffle
import numpy as np

import hands_dataset

def endless_cropped_masked_hand_animations(np_data):
    while True:
        idxs = list(range(len(np_data)))
        shuffle(idxs)
        for i in idxs:
            filename, n_frames, data, is_right_hand = np_data[i]
            window_size = 100
            n_dof = 23
            data = data * np.pi / 360. # convert to radians
            window_start = torch.randint(low=0, high=n_frames-window_size, size=[])
            chunk = data[window_start:window_start+window_size]
            frame_idxs = torch.range(window_start, window_start+window_size).reshape([window_size, 1, 1]).repeat(1, n_dof, 1)
            dof_idxs = torch.arange(n_dof).reshape([1, n_dof, 1]).repeat(window_size, 1, 1)
            is_right_hand = torch.tensor(is_right_hand).reshape([1, 1, 1]).repeat(window_size, n_dof, 1)

            yield {
                'angles': chunk,
                'frame_idxs': frame_idxs,
                'dof_idxs': dof_idxs,
                'hand_idxs': is_right_hand,
            }

class HandsDataset(torch.utils.data.Dataset):
    def __init__(self, bvh_dir):
        self.np_data = hands_dataset.np_dataset(bvh_dir="./BVH")

    def __iter__(self):
        return endless_cropped_masked_hand_animations(self.np_data)
