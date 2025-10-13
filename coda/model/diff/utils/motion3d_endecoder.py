import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, einsum, repeat
from coda.network.evaluator.statistics import T2M_VEC, ARCTIC_VEC, ARCTICOBJ_VEC, GRAB_OBJVEC, GRAB_OBJVEC_NOROT


class HmlvecArcticEnDecoder(nn.Module):
    def __init__(self):
        """Original implementation"""
        super().__init__()
        self.register_buffer("mean", torch.tensor(ARCTIC_VEC["mean"]).float(), False)
        self.register_buffer("std", torch.tensor(ARCTIC_VEC["std"]).float(), False)
        self.mean = self.mean.reshape(-1)
        self.std = self.std.reshape(-1)

    def encode(self, data):
        """
        Args:
            data: (B, L, J, 3), in ayfz coordinate
            length: (B,), effective length of each sample
        Returns:
            x: (B, 263, L)
        """
        data = data.clone()
        B, L, _, _ = data.shape
        vec = data.reshape(B, L, -1)
        x = (vec - self.mean) / self.std

        return x.permute(0, 2, 1)


class HmlobjvecArcticEnDecoder(nn.Module):
    def __init__(self):
        """Original implementation"""
        super().__init__()
        self.register_buffer("mean", torch.tensor(ARCTICOBJ_VEC["mean"]).float(), False)
        self.register_buffer("std", torch.tensor(ARCTICOBJ_VEC["std"]).float(), False)
        self.mean = self.mean.reshape(-1)
        self.std = self.std.reshape(-1)

    def encode(self, data):
        """
        Args:
            data: (B, L, J, 3), in ayfz coordinate
            length: (B,), effective length of each sample
        Returns:
            x: (B, 263, L)
        """
        vec = data.clone()
        x = (vec - self.mean) / self.std

        return x.permute(0, 2, 1)


class HmlobjvecGrabEnDecoder(nn.Module):
    def __init__(self):
        """Original implementation"""
        super().__init__()
        self.register_buffer("mean", torch.tensor(GRAB_OBJVEC["mean"]).float(), False)
        self.register_buffer("std", torch.tensor(GRAB_OBJVEC["std"]).float(), False)

        # self.register_buffer("mean", torch.tensor(GRAB_OBJVEC_NOROT["mean"]).float(), False)
        # self.register_buffer("std", torch.tensor(GRAB_OBJVEC_NOROT["std"]).float(), False)
        self.mean = self.mean.reshape(-1)
        self.std = self.std.reshape(-1)

        # self.std[...] = 1.0

    def encode(self, data):
        """
        Args:
            data: (B, L, J, 3), in ayfz coordinate
            length: (B,), effective length of each sample
        Returns:
            x: (B, 263, L)
        """
        vec = data.clone()
        x = (vec - self.mean) / self.std

        return x.permute(0, 2, 1)
