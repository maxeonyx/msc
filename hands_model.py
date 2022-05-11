import torch
from torch import nn

class HandsModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        n_frames = 100
        n_embd = 256
        n_dof = 23
        n_hands = 2

        self.frame_embedding = nn.Embedding(n_frames, n_embd)
        self.dof_embedding = nn.Embedding(n_dof, n_embd)
        self.hand_embedding = nn.Embedding(n_hands, n_embd)
        self.encoder = nn.Linear(1, n_embd, )
        self.transformer = nn.TransformerE(
            d_model=n_embd,
            dim_feedforward=512,
            num_decoder_layers=4
        )
