from jinja2 import pass_context
import torch
from torch import nn, Tensor
import math

class CustomPositionalEncoding(nn.Module):
    """
    Used to encode angles as vectors. Since we want to maintain similarity across the boundary, always multiply
    the inputs by an integer before passing them through the sin and cos. This means they smoothly wrap.
    """

    def __init__(self, d_embd: int, base=2, seq_range: float = math.tau):
        super().__init__()
        self.d_embd = d_embd
        self.div_term = torch.arange(start=1, end=d_embd//2+1)
        self.div_term = self.div_term.unsqueeze(0).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.unsqueeze(-1)
        x_new = torch.zeros_like(x).repeat(1, 1, self.d_embd)
        print(self.div_term.shape)
        print(x.shape)
        print(x_new.shape)
        x_new[:, :, 0::2] = torch.sin(x * self.div_term)
        x_new[:, :, 1::2] = torch.cos(x * self.div_term)
        return x_new

class HandsTransformerModel(nn.Module):

    def __init__(self, n_frames, n_dof, n_hands, ) -> None:
        super().__init__()

        n_frames = 100
        d_embd = 256
        n_dof = 23
        n_hands = 2

        self.angle_embedding = CustomPositionalEncoding(d_embd)
        self.frame_embedding = nn.Embedding(n_frames, d_embd)
        self.dof_embedding = nn.Embedding(n_dof, d_embd)
        self.hand_embedding = nn.Embedding(n_hands, d_embd)

        self.encoder = nn.Linear(1, d_embd)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embd,
            nhead=8,
            dim_feedforward=512,

        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3,
        )
        self.head = nn.Linear(d_embd, 1)


    def forward(self, angles: torch.Tensor, frame_idxs, dof_idxs, hand_idxs):

        frame_idxs = self.frame_embedding(frame_idxs)
        dof_idxs = self.dof_embedding(dof_idxs)
        hand_idxs = self.hand_embedding(hand_idxs)
        angles = self.angle_embedding(angles)
        
        inp = angles + frame_idxs + dof_idxs + hand_idxs
