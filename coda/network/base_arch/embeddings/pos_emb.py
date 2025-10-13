import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (L, C)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (L, 1, C)

        self.register_buffer("pe", pe)

    def forward(self, x, is_BLC=False):
        # L, B, C
        # not used in the final model
        if is_BLC:
            x = x + self.pe.transpose(0, 1)[:, : x.shape[1]]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps, is_BLC=False):
        if is_BLC:
            # return (B, 1, C)
            return self.time_embed(self.sequence_pos_encoder.pe[timesteps])
        else:
            # return (1, B, C)
            return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
